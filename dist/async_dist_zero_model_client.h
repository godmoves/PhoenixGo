#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stack>
#include <thread>

#include "model/zero_model_base.h"

#include "dist/async_rpc_queue.h"
#include "dist/dist_config.pb.h"
#include "dist/dist_zero_model.grpc.pb.h"
#include "dist/leaky_bucket.h"

class AsyncDistZeroModelClient final : public ZeroModelBase {
public:
  AsyncDistZeroModelClient(const std::vector<std::string> &svr_addrs,
                           const DistConfig &dist_config);

  ~AsyncDistZeroModelClient() override;

  int Init(const ModelConfig &model_config) override;

  int Forward(const std::vector<std::vector<bool>> &inputs,
              std::vector<std::vector<float>> &policy,
              std::vector<float> &value) override;

  void Forward(const std::vector<std::vector<bool>> &inputs,
               callback_t callback) override;

  int GetGlobalStep(int &global_step) override;

  int RpcQueueSize() override;

  void Wait() override;

private:
  int GetStub();

  void ReleaseStub(int stub_id);

  void DisableStub(int stub_id);

private:
  DistConfig m_config;
  std::vector<std::string> m_svr_addrs;
  std::vector<std::unique_ptr<DistZeroModel::Stub>> m_stubs;
  AsyncRpcQueue m_forward_rpc_queue;
  std::thread m_forward_rpc_complete_thread;

  std::stack<int> m_avail_stubs;
  std::mutex m_mutex;
  std::condition_variable m_cond;

  std::vector<LeakyBucket> m_leaky_buckets;
  std::unique_ptr<std::mutex[]> m_bucket_mutexes;
};
