package com.example.tflite_nare

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.debugInspectorInfo
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.lifecycle.LifecycleOwner
import com.example.tflite_nare.ui.theme.TFlite_NareTheme
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

const val TAG = "Nomal"
const val TAG_Camera = "Camera"
const val TAG_TF = "TFlite_Gahi"

lateinit var imageCapture:ImageCapture
var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
lateinit var previewView:PreviewView
lateinit var lifecycleOwner:LifecycleOwner
lateinit var backgroundExecutor:ExecutorService
var toast: Toast? = null
//lateinit var outputOptions:ImageCapture.OutputFileOptions
data class Marker(var x:Double = 0.0, var y:Double = 0.0, var z:Double = 0.0)

private val permissionList = listOf(
    android.Manifest.permission.CAMERA,
    android.Manifest.permission.BLUETOOTH,
    android.Manifest.permission.BLUETOOTH_CONNECT,
    android.Manifest.permission.BLUETOOTH_ADMIN,
    android.Manifest.permission.BLUETOOTH_SCAN,
    android.Manifest.permission.ACCESS_COARSE_LOCATION,
)
class MainActivity : ComponentActivity(), SegmentationPipeline.SegmentationListener {
    companion object {
        private const val ALPHA_COLOR = 128
    }

    lateinit var seg:SegmentationPipeline
    val SegmentationResult:MutableState<List<Segmentation>> = mutableStateOf(listOf())
    fun checkPermission() {
        permissionList.forEach {
            val cameraPermission = ContextCompat.checkSelfPermission(
                this@MainActivity,
                it
            )
            if(cameraPermission != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissionList.toTypedArray(), 1000)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkPermission()
        seg = SegmentationPipeline(
        context = this,
        imageSegmentationListener = this,
            numThreads = 3,
            currentDelegate = SegmentationPipeline.DELEGATE_CPU
        )

        setContent {
            TFlite_NareTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Box(
                        contentAlignment = Alignment.TopCenter,
                        modifier = Modifier
                            .padding(top = 50.dp)

                    ) {
                        CameraView(
                            Modifier
                                .size(300.dp, 400.dp)
//                                    .fillMaxWidth()
//                                    .height(300.dp)
                                .border(3.dp, Color.Cyan))
                    }
                }
            }
        }
    }
    @Composable
    fun CameraView(modifier:Modifier) {
        lifecycleOwner = LocalLifecycleOwner.current
        Surface(modifier) {
            backgroundExecutor = Executors.newSingleThreadExecutor()
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { context ->
                    previewView = PreviewView(context).apply {
                        this.scaleType = scaleType
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.FILL_PARENT,
                            ViewGroup.LayoutParams.FILL_PARENT
                        )
                    }
                    startCam(context, previewView, lifecycleOwner)
                    previewView
                }
            )
            landmarker(modifier = Modifier.fillMaxSize())

        }
    }
    @Composable
    fun landmarker(modifier: Modifier) {
        Canvas(modifier = modifier, onDraw =
        {
            if (SegmentationResult.value.isNotEmpty()) {
                val colorLabels =
                    SegmentationResult.value[0].coloredLabels.mapIndexed { index, coloredLabel ->
                        ColorLabel(
                            index,
                            coloredLabel.getlabel(),
                            coloredLabel.argb
                        )
                    }
                val maskTensor = SegmentationResult.value[0].masks[0]
                val maskArray = maskTensor.buffer.array()
                val pixels = IntArray(maskArray.size)

                for (i in maskArray.indices) {

                    val colorLabel = colorLabels[maskArray[i].toInt()].apply {
                        isExist = true
                    }
                    val color = colorLabel.getColor()
                    pixels[i] = color
//                    if(color.red > 0f) {
//                    }
//                    Log.d("Colors", "->| ${0xFF6650a4.toInt()} || ${color}")


                }
//                Log.d("Colors", "->| ${0xFF6650a4.toInt()} || ${Color.Yellow.hashCode()}")
                val matrix: Matrix = Matrix()
                matrix.setScale(-1f, 1f)
                val source = Bitmap.createBitmap(
                    pixels,
                    maskTensor.width,
                    maskTensor.height,
                    Bitmap.Config.ARGB_8888
                )
                val image = Bitmap.createBitmap(
                    source,
                    0,
                    0,
                    maskTensor.width,
                    maskTensor.height,
                    matrix,
                    true
                )

//                Log.d("Colors", "->| ${image.height} ${pixels.size}")


                // PreviewView is in FILL_START mode. So we need to scale up the bounding
                // box to match with the size that the captured images will be displayed.
                val scaleFactor = max(drawContext.size.width * 1f / 480f, drawContext.size.height * 1f / 640f)
                val scaleWidth = (480f * scaleFactor).toInt()
                val scaleHeight = (640f * scaleFactor).toInt()

                val scaleBitmap = Bitmap.createScaledBitmap(image, scaleWidth, scaleHeight, false)
                image.recycle()

//                Log.d("Colors", "${image.getColor(100,100)} | ${scaleWidth}, ${scaleHeight}")

                drawImage(
                    scaleBitmap.asImageBitmap(),
                )
            }
//            drawCircle(Color.Cyan, 100f, Offset(drawContext.size.width, drawContext.size.height))
        })
    }
    @SuppressLint("UnsafeOptInUsageError")
    fun startCam(context: Context, previewView: PreviewView, lifecycleOwner: LifecycleOwner) {



        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

//            val cam2Infos = cameraProvider.availableCameraInfos.map {
//                Camera2CameraInfo.from(it)
//            }.sortedByDescending {
//                // HARDWARE_LEVEL is Int type, with the order of:
//                // LEGACY < LIMITED < FULL < LEVEL_3 < EXTERNAL
//                it.getCameraCharacteristic(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
//            }
//            cam2Infos.forEach{
//                Log.e("Cam2Info", it.cameraId)
//            }
            // Preview
            val preview = androidx.camera.core.Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()
            val imageAnalyzer =
                ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
//                    .setTargetResolution(Size(725, 1000))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    // The analyzer can then be assigned to the instance
                    .also {
                        it.setAnalyzer(backgroundExecutor) { image ->
                            val bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )

                            image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
                            seg.segment(bitmapBuffer, image.imageInfo.rotationDegrees)

                        }
                    }
            fun selectExternalOrBestCamera(provider: ProcessCameraProvider):CameraSelector? {
                val cam2Infos = provider.availableCameraInfos.map {
                    Camera2CameraInfo.from(it)
                }.sortedByDescending {
                    // HARDWARE_LEVEL is Int type, with the order of:
                    // LEGACY < LIMITED < FULL < LEVEL_3 < EXTERNAL
                    it.getCameraCharacteristic(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                }

                return when {
                    cam2Infos.isNotEmpty() -> {
                        CameraSelector.Builder()
                            .addCameraFilter {
                                it.filter { camInfo ->
                                    // cam2Infos[0] is either EXTERNAL or best built-in camera
                                    val thisCamId = Camera2CameraInfo.from(camInfo).cameraId
                                    thisCamId == cam2Infos[1].cameraId
//                                    thisCamId == cam2Infos[3].cameraId
                                }
                            }.build()
                    }
                    else -> null
                }
            }


//            val asas = HandLandmarkerHelper(context = ))
//            asas.detectLiveStream()
            // Select back camera as a default
//                    Log.e("Kamera", "${android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT}")
            try {
                val selector = selectExternalOrBestCamera(cameraProvider)

                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, selector!!, preview, imageCapture, imageAnalyzer
                )


            } catch(exc: Exception) {
                Log.e("Kamel", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    override fun onError(error: String) {
        TODO("Not yet implemented")
    }

    override fun onResults(
        results: List<Segmentation>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        if (results != null) {
            SegmentationResult.value = results
            Log.e("Camera", results[0].masks.size.toString())

        }
    }

    data class ColorLabel(
        val id: Int,
        val label: String,
        val rgbColor: Int,
        var isExist: Boolean = false
    ) {

        fun getColor(): Int {
            // Use completely transparent for the background color.

            return if (id == 0) Color.hashCode() else Color(
                rgbColor%256,
                (rgbColor/256)%256,
                (rgbColor/65536)%256,
                0x97
            ).hashCode()
        }
    }
}

enum class ButtonState { Pressed, Idle }
fun Modifier.bounceClick() = composed {
    /**
     * https://blog.canopas.com/jetpack-compose-cool-button-click-effects-c6bbecec7bcb
     * 위 글을 참고 하였다.
     **/
    var buttonState by remember { mutableStateOf(ButtonState.Idle) }
    val scale by animateFloatAsState(if (buttonState == ButtonState.Pressed) 0.70f else 1f,
        label = ""
    )

    this
        .graphicsLayer {
            scaleX = scale
            scaleY = scale
        }
        .clickable(
            interactionSource = remember { MutableInteractionSource() },
            indication = null,
            onClick = { }
        )
        .pointerInput(buttonState) {
            awaitPointerEventScope {
                buttonState = if (buttonState == ButtonState.Pressed) {
                    waitForUpOrCancellation()
                    ButtonState.Idle
                } else {
                    awaitFirstDown(false)
                    ButtonState.Pressed
                }
            }
        }
}

//fun Modifier.MixClick(
//    enabled: Boolean = true,
//    onClickLabel: String? = null,
//    role: Role? = null,
//    interactionSource: MutableInteractionSource,
//    onClick: () -> Unit
//) = composed(
//    inspectorInfo = debugInspectorInfo {
//        name = "clickable"
//        properties["enabled"] = enabled
//        properties["onClickLabel"] = onClickLabel
//        properties["role"] = role
//        properties["onClick"] = onClick
//    }
//) {
//    Modifier.clickable(
//        enabled = enabled,
//        onClickLabel = onClickLabel,
//        onClick = onClick,
//        role = role,
//        indication = LocalIndication.current,
//        interactionSource = interactionSource
//    )
//}
@OptIn(ExperimentalFoundationApi::class)
fun Modifier.MixClick(
    enabled: Boolean = true,
    onClickLabel: String? = null,
    role: Role? = null,
    onLongClickLabel: String? = null,
    onLongClick: (() -> Unit)? = null,
    onDoubleClick: (() -> Unit)? = null,
    interactionSource: MutableInteractionSource,
    onClick: () -> Unit,
) = composed(
    inspectorInfo = debugInspectorInfo {
        name = "combinedClickable"
        properties["enabled"] = enabled
        properties["onClickLabel"] = onClickLabel
        properties["role"] = role
        properties["onClick"] = onClick
        properties["onDoubleClick"] = onDoubleClick
        properties["onLongClick"] = onLongClick
        properties["onLongClickLabel"] = onLongClickLabel
    }
) {
    Modifier.combinedClickable(
        enabled = enabled,
        onClickLabel = onClickLabel,
        onLongClickLabel = onLongClickLabel,
        onLongClick = onLongClick,
        onDoubleClick = onDoubleClick,
        onClick = onClick,
        role = role,
        indication = LocalIndication.current,
        interactionSource = interactionSource
    )
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    TFlite_NareTheme {
        Greeting("Android")
    }
}