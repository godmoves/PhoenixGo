#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) override
    {
        switch (severity) {
          case Severity::kINTERNAL_ERROR: std::cout << "kINTERNAL_ERROR: " << msg << "\n"; break;
          case Severity::kERROR: std::cout << "kERROR: " << msg << "\n"; break;
          case Severity::kWARNING: std::cout << "kWARNING: " << msg << "\n"; break;
          case Severity::kINFO: std::cout << "kINFO: " << msg << "\n"; break;
        }
    }
} g_logger;


int main() {
    const int INPUT_DIM = 18 * 19 * 19;
    const int OUTPUT_DIM = 362;

    std::string tensorrt_model_path = "../../leelaz-model-0.PLAN";
    // std::string tensorrt_model_path = "";
    std::string uff_model_name = "../../leelaz-model-0.uff";

    nvinfer1::IBuilder *builder;
    nvinfer1::INetworkDefinition *network;
    nvuffparser::IUffParser *parser;
    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IRuntime *m_runtime;
    nvinfer1::IExecutionContext *m_context;
    std::vector<void*> m_cuda_buf;

    if (tensorrt_model_path == "") {
        // using uff file when PLAN is not available
        std::cout << "INFO: using uff model\n";

        builder = nvinfer1::createInferBuilder(g_logger);
        network = builder->createNetwork();
        parser = nvuffparser::createUffParser();

        parser->registerInput("inputs", nvinfer1::DimsCHW(18, 19, 19), nvuffparser::UffInputOrder::kNCHW);
        parser->registerOutput("policy");
        parser->registerOutput("value");

        // kFLOAT or kHALF
        nvinfer1::DataType m_datatype = nvinfer1::DataType::kHALF;

        if (!parser->parse(uff_model_name.c_str(), *network, m_datatype)) {
            std::cout << "Failed to parse UFF\n";
            builder->destroy();
            network->destroy();
            parser->destroy();
            return 1;
        }

        // build engine
        if (m_datatype == nvinfer1::DataType::kHALF)
            builder->setHalf2Mode(true);


        builder->setMaxBatchSize(4);
        builder->setMaxWorkspaceSize(1 << 20);
        m_engine = builder->buildCudaEngine(*network);

        // nvinfer1::IHostMemory *serializedEngine = engine->serialize();
    } else {
        std::cout << "INFO: using PLAN model\n";

        std::ostringstream model_ss(std::ios::binary);
        if (!(model_ss << std::ifstream(tensorrt_model_path, std::ios::binary).rdbuf())) {
            std::cout << "ERROR: read tensorrt model '" << tensorrt_model_path << "' error\n";
            return 1;
        }
        std::string model_str = model_ss.str();

        m_runtime = nvinfer1::createInferRuntime(g_logger);
        m_engine = m_runtime->deserializeCudaEngine(model_str.c_str(), model_str.size(), nullptr);
        if (m_engine == nullptr) {
            std::cout << "ERROR: load cuda engine error\n";
            return 1;
        }
    }

    std::vector<float> feature(19 * 19 * 18, 0);
    for (int i = 0; i < 18; ++i) {
        for (int j = 0; j < 19; ++j) {
            for (int k = 0; k < 19; ++k) {
                if (i == 16)
                    feature[19*19*i + 19*j + k] = 1;
            }
        }
    }
    std::vector<std::vector<float>> inputs;
    inputs.push_back(feature);

    m_context = m_engine->createExecutionContext();

    int max_batch_size = m_engine->getMaxBatchSize();
    std::cout << "INFO: tensorrt max batch size: " << max_batch_size <<"\n";
    for (int i = 0; i < m_engine->getNbBindings(); ++i) {
        auto dim = m_engine->getBindingDimensions(i);
        std::string dim_str = "(";
        int size = 1;
        for (int i = 0; i < dim.nbDims; ++i) {
            if (i) dim_str += ", ";
            dim_str += std::to_string(dim.d[i]);
            size *= dim.d[i];
        }
        dim_str += ")";
        std::cout << "INFO: tensorrt binding: " << m_engine->getBindingName(i) << " " << dim_str <<"\n";

        void *buf;
        int ret = cudaMalloc(&buf, max_batch_size * size * sizeof(float));
        if (ret != 0) {
            std::cout << "ERROR: cuda malloc err " << ret <<"\n";
            return 1;
        }
        m_cuda_buf.push_back(buf);
    }

    int batch_size = inputs.size();

    std::vector<float> inputs_flat(batch_size * INPUT_DIM);
    for (int i = 0; i < batch_size; ++i) {
        if (inputs[i].size() != INPUT_DIM) {
            std::cout << "ERROE: Error input dim not match, need " << INPUT_DIM << ", got " << inputs[i].size() << "\n";
            return 1;
        }
        for (int j = 0; j < INPUT_DIM; ++j) {
            inputs_flat[i * INPUT_DIM + j] = inputs[i][j];
        }
    }

    int ret = cudaMemcpy(m_cuda_buf[0], inputs_flat.data(), inputs_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (ret != 0) {
        std::cout << "ERROR: cuda memcpy err " << ret << "\n";
        return 1;
    }

    m_context->execute(batch_size, m_cuda_buf.data());

    std::vector<std::vector<float>> policy;
    std::vector<float> value;

    std::vector<float> policy_flat(batch_size * OUTPUT_DIM);
    ret = cudaMemcpy(policy_flat.data(), m_cuda_buf[1], policy_flat.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (ret != 0) {
        std::cout << "ERROR: cuda memcpy err " << ret << "\n";
        return 1;
    }
    policy.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        policy[i].resize(OUTPUT_DIM);
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            policy[i][j] = policy_flat[i * OUTPUT_DIM + j];
        }
    }

    value.resize(batch_size);
    ret = cudaMemcpy(value.data(), m_cuda_buf[2], value.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (ret != 0) {
        std::cout << "ERROR: cuda memcpy err " << ret << "\n";
        return 1;
    }
    for (int i = 0; i < batch_size; ++i) {
        value[i] = -value[i];
    }

    std::cout << "policy head:\n";
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            printf("%.4f ", policy[0][19*i + j]);
        }
        printf("\n");
    }
    std::cout << "value head:\n" << value[0] << "\n";

    std::cout << "Done\n";
    return 0;
}
