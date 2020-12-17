# shim::StatusOr

It is a StatusOr re-implementation of TensorFlow StatusOr that uses absl::Status
instead of TensorFlow Status. This is a temporary solution and it will be
dropped as soon as absl::StatusOr becomes available in Abseil.

Reliance on `shim` namespace may require changing your code to
support absl::StatusOr instead of shim::StatusOr.
