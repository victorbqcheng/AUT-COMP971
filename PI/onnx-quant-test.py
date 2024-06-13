import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16


model_fp32 = './yolov8n.onnx'
model_quant = './yolov8n.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QInt8)
