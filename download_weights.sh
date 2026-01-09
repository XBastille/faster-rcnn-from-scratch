#!/bin/bash

mkdir -p outputs

echo "Downloading best.pth..."
curl -L "https://github.com/XBastille/faster-rcnn-from-scratch/releases/download/v1/best.pth" -o outputs/best.pth

echo "Downloading model.onnx..."
curl -L "https://github.com/XBastille/faster-rcnn-from-scratch/releases/download/v1/model.onnx" -o outputs/model.onnx

echo "Done! Weights saved to outputs/"
