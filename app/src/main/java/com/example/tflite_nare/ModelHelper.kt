package com.example.tflite_nare

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.create
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.DelegateFactory
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel


class ModelHelper(
    val luna: Luna,
    val assetManager: AssetManager,
) {
    lateinit var inter:InterpreterApi
    private lateinit var inputImage : TensorImage
    private var modelInputWidths:Int = 0
    private var modelInputHeights:Int = 0
    private lateinit var result:TensorBuffer
    private lateinit var outputBuffer: TensorBuffer
    init {
        setModel()
    }
    fun setModel() {
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            Log.e("tftf", compatList.isDelegateSupportedOnThisDevice.toString())
            if (compatList.isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
//                val nnApiDelegate = NnApiDelegate()
//                this.addDelegate(nnApiDelegate)
            }

        }

        inter = create(loadModelFile(), options)

//        var nnApiDelegate: NnApiDelegate? = null
//        var GpuDelegate: GpuDelegate? = null
//        nnApiDelegate = NnApiDelegate()
//        GpuDelegate = GpuDelegate()
//        options.addDelegate(GpuDelegate)
//        options.numThreads = 5
//        options.addDelegate {
//            DELEGATE_NNAPI.toLong()
//        }
//        options.useNNAPI = true

        val inputTensor = inter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        inputImage = TensorImage(inputTensor.dataType())
        val modelInputChannel = inputShape[0]
        val modelInputWidth = inputShape[1]
        val modelInputHeight = inputShape[2]

        modelInputHeights = modelInputWidth
        modelInputWidths = modelInputHeight

        val outputTensor = inter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

        Log.d("Model Info", "| InPut | ${inputTensor.shape().toList()}, ${inputTensor.dataType()} |  | OutPut | ${outputShape.toList()}, ${outputTensor.dataType()} | <--|")
    }

    private fun loadImage(bitmap: Bitmap) : TensorImage {
        inputImage.load(bitmap)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputHeights, modelInputWidths, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
//            .add(NormalizeOp(0.0f, 255.0f))
            .build()
        return imageProcessor.process(inputImage)
    }

    fun Result(img:Bitmap) {
        inputImage = loadImage(img)
        var inferenceTime = SystemClock.uptimeMillis()

        inter.run(inputImage.buffer, outputBuffer.buffer)

        val DataType = DataType.FLOAT32 // 입력 데이터 타입
        val gahi = TensorBufferFloat.createFrom(outputBuffer, DataType)

//        Log.e("result", gahi.floatArray.max().toString())
//        Log.e("result", img.getPixel(50,50).toString())
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        luna.onResult(gahi.floatArray, inferenceTime)
    }


    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assetManager.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        Log.e("tftf", "tftf")
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    interface Luna{
        fun onResult(
            results: FloatArray,
            inferenceTime: Long
        )
    }
    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
//        const val modelPath = "deeplabv3.tflite"
//        const val modelPath = "model.tflite"
        const val modelPath = "Depther.tflite"

        private const val TAG = "LUNA"
    }
}

