#pragma once

#include <boost/filesystem.hpp>

boost::filesystem::path GetCheckpointPath(const boost::filesystem::path &model_dir);

bool CopyCheckpoint(const boost::filesystem::path &from, const boost::filesystem::path &to);
