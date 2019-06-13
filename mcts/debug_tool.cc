#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/go_state.h"
#include "common/timer.h"
#include "model/trt_zero_model.h"
#include "model/tf_zero_model.h"
#include "model/tpu_zero_model.h"

#include "mcts_config.h"

DEFINE_string(config_path, "", "Path of mcts config file.");
DEFINE_string(init_moves, "", "Initialize Go board with init_moves.");
DEFINE_int32(gpu, 0, "gpu used by neural network.");
DEFINE_int32(intra_op_parallelism_threads, 0, "Number of tf's intra op threads");
DEFINE_int32(inter_op_parallelism_threads, 0, "Number of tf's inter op threads");
DEFINE_int32(transform, 0, "Transform features.");
DEFINE_int32(num_iterations, 1, "How many iterations should run.");
DEFINE_int32(batch_size, 1, "Batch size of each iterations.");
DEFINE_bool(show_features, false, "Show input features");

void InitMove(GoState &board, std::string &moves) {
  for (size_t i = 0; i < moves.size(); i += 3) {
    int x = -1, y = -1;
    if (moves[i] != 'z') {
      x = moves[i] - 'a';
      y = moves[i + 1] - 'a';
    }
    board.Move(x, y);
  }
}

void TransformCoord(GoCoordId &x, GoCoordId &y, int mode, bool reverse) {
  if (reverse) {
    if (mode & 4)
      std::swap(x, y);
    if (mode & 2)
      y = GoComm::BOARD_SIZE - y - 1;
    if (mode & 1)
      x = GoComm::BOARD_SIZE - x - 1;
  } else {
    if (mode & 1)
      x = GoComm::BOARD_SIZE - x - 1;
    if (mode & 2)
      y = GoComm::BOARD_SIZE - y - 1;
    if (mode & 4)
      std::swap(x, y);
  }
}

template <class T>
void TransformFeatures(T &features, int mode, bool reverse) {
  T ret(features.size());
  int depth = features.size() / GoComm::BOARD_INTERSECTIONS;
  for (int i = 0; i < GoComm::BOARD_INTERSECTIONS; ++i) {
    GoCoordId x, y;
    GoFunction::IdToCoord(i, x, y);
    TransformCoord(x, y, mode, reverse);
    int j = GoFunction::CoordToId(x, y);
    for (int k = 0; k < depth; ++k) {
      ret[i + k * GoComm::BOARD_INTERSECTIONS] = features[j + k * GoComm::BOARD_INTERSECTIONS];
    }
  }
  features = std::move(ret);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  auto config = LoadConfig(FLAGS_config_path);
  CHECK(config != nullptr) << "Load mcts config file '" << FLAGS_config_path << "' failed";

  if (FLAGS_intra_op_parallelism_threads > 0) {
    config->mutable_model_config()->set_intra_op_parallelism_threads(
        FLAGS_intra_op_parallelism_threads);
  }

  if (FLAGS_inter_op_parallelism_threads > 0) {
    config->mutable_model_config()->set_inter_op_parallelism_threads(
        FLAGS_inter_op_parallelism_threads);
  }

  if (config->model_config().enable_mkl()) {
    TfZeroModel::SetMKLEnv(config->model_config());
  }

  std::unique_ptr<ZeroModelBase> model(new TfZeroModel(FLAGS_gpu));
  if (config->model_config().enable_tensorrt()) {
    model.reset(new TrtZeroModel(FLAGS_gpu));
  }
  if (config->model_config().enable_tpu()) {
    model.reset(new TpuZeroModel(FLAGS_gpu));
  }

  CHECK_EQ(model->Init(config->model_config()), 0)
      << "Model Init Fail, config path " << FLAGS_config_path
      << ", gpu " << FLAGS_gpu;

  GoState board;
  InitMove(board, FLAGS_init_moves);

  auto features = board.GetFeature();
  if (FLAGS_show_features) {
    for (int f = 0; f < GoFeature::FEATURE_COUNT; ++f) {
      printf("Feature plane %d\n", f);
      for (int i = 0; i < GoComm::BOARD_SIZE; ++i) {
        for (int j = 0; j < GoComm::BOARD_SIZE; ++j) {
          int id = f * GoComm::BOARD_INTERSECTIONS +
                   i * GoComm::BOARD_SIZE +
                   j;
          if (features[id]) {
            printf("X ");
          } else {
            printf(". ");
          }
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  TransformFeatures(features, FLAGS_transform, false);
  std::vector<std::vector<bool>> inputs(FLAGS_batch_size, features);

  std::vector<std::vector<float>> policies;
  std::vector<float> values;

  Timer timer;
  for (int i = 1; i <= FLAGS_num_iterations; ++i) {
    CHECK_EQ(model->Forward(inputs, policies, values), 0) << "Forward fail";
    LOG_IF(INFO, i % 100 == 0) << i << "/" << FLAGS_num_iterations << " iterations";
  }
  float avg_cost_ms = timer.fms() / FLAGS_num_iterations;
  LOG(INFO) << "Cost " << avg_cost_ms << "ms per iteration";

  std::vector<float> &policy = policies[0];
  TransformFeatures(policy, FLAGS_transform, true);
  float value = values[0];
  board.ShowBoard();
  for (int i = 0; i < GoComm::BOARD_SIZE; ++i) {
    for (int j = 0; j < GoComm::BOARD_SIZE; ++j) {
      printf("%.4f ", policy[i * GoComm::BOARD_SIZE + j]);
    }
    puts("");
  }
  printf("Value %.4f Pass %.4f\n", value, policy[361]);

  return 0;
}
