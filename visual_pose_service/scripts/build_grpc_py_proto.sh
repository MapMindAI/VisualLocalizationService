#!/bin/bash
set -e


bazel build //proto/map:visual_pose_service_py_proto
bazel build //proto/map:visual_pose_service_py_grpc

rm -rf /Mobili/python/visual_pose_service/proto
cp -r /Mobili/bazel-bin/proto/map/visual_pose_service_py_grpc_pb/proto /Mobili/python/visual_pose_service/proto
