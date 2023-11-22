package com.example.tflite_nare

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.*
import org.tensorflow.lite.InterpreterApi.create
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel


class ModelHelper(
    val luna: Luna,
    val assetManager: AssetManager,
) {
    lateinit var inter:InterpreterApi

    init {
        setModel()
    }
    fun setModel() {
        val options = InterpreterApi.Options()
        options.numThreads = 3

        options.setNumThreads(3)
//        options.useNNAPI = true

        val input = Array(224) {Array(224) {FloatArray(3){0f}}}


        inter = create(loadModelFile(), options)

        val inputTensor = inter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val modelInputChannel = inputShape[0]
        val modelInputWidth = inputShape[1]

        val outputTensor = inter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val modelOutputClasses = outputShape[1]
        Log.d("Model Info", "| InPut | ${inputTensor.shape().toList()}, ${inputTensor.dataType()} |  | OutPut | ${outputShape.toList()}, ${outputTensor.dataType()} | <--|")
    }

    fun Result(img:Bitmap) {
        val inputSize = 224
        val inputChannels = 3

        // 이미지를 224x224 크기로 변경
        val resizedBitmap = Bitmap.createScaledBitmap(img, inputSize, inputSize, true)
        Log.e("img", resizedBitmap.width.toString() + "||" + resizedBitmap.height.toString())

        val byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * inputChannels * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        resizedBitmap.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()

        // ByteBuffer를 TensorBuffer로 변환하여 TFLite 모델에 전달할 텐서 생성
        val inputDataType = DataType.FLOAT32 // 입력 데이터 타입
        val shape = intArrayOf(1, 224, 224, 3)
        val tensorBuffer = TensorBuffer.createFixedSize(shape, inputDataType)
        tensorBuffer.loadBuffer(byteBuffer)

//        val result = Array(224) {Array(224) {FloatArray(3){0f}}}


        val DataType = DataType.FLOAT32 // 입력 데이터 타입
        val shapes = intArrayOf(1, 224, 224, 1)
        val result = TensorBuffer.createFixedSize(shapes, DataType)
        Log.e("Oresult", result.buffer.array().size.toString())

        var inferenceTime = SystemClock.uptimeMillis()

        inter.run(tensorBuffer.buffer, result.buffer)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        Log.e("result", result.floatArray.size.toString())

        luna.onResult(result, inferenceTime)
    }


    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assetManager.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    interface Luna{
        fun onResult(
            results: TensorBuffer,
            inferenceTime: Long
        )
    }
    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val modelPath = "model.tflite"

        private const val TAG = "LUNA"
    }
}
