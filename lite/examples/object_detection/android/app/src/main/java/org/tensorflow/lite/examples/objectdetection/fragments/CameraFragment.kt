
package org.tensorflow.lite.examples.objectdetection.fragments

import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(),
    ObjectDetectorHelper.DetectorListener {

    companion object {
        private const val TAG = "ObjectDetection"

        private const val REQUIRED_STABLE_FRAMES = 3
        private const val MIN_ANNOUNCEMENT_GAP_MS = 2000L
        private const val SAME_LABEL_REPEAT_DELAY_MS = 6000L
        private const val SPEECH_CONFIDENCE_THRESHOLD = 0.60f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding: FragmentCameraBinding
        get() = _fragmentCameraBinding
            ?: throw IllegalStateException(
                "Camera binding is unavailable"
            )

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var cameraExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var preview: Preview? = null

    private var textToSpeech: TextToSpeech? = null
    private var ttsReady = false

    private var candidateLabel = ""
    private var candidateCount = 0

    private var lastAnnouncedLabel = ""
    private var lastAnnouncementTime = 0L
    private var lastLabelAnnouncementTime = 0L

    private var utteranceCounter = 0L

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(
                inflater,
                container,
                false
            )

        return fragmentCameraBinding.root
    }

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?
    ) {
        super.onViewCreated(view, savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext().applicationContext,
            objectDetectorListener = this
        )

        initializeTextToSpeech()

        // Hide visual overlays for audio-first assistance.
        fragmentCameraBinding.overlay.visibility = View.GONE

        fragmentCameraBinding.viewFinder.post {
            if (_fragmentCameraBinding != null) {
                startCamera()
            }
        }
    }

    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(
            requireContext().applicationContext
        ) { status ->

            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.US)

                ttsReady =
                    result != TextToSpeech.LANG_MISSING_DATA &&
                    result != TextToSpeech.LANG_NOT_SUPPORTED &&
                    result != null

                if (!ttsReady) {
                    Log.e(TAG, "Text-to-speech language unavailable")
                }
            } else {
                ttsReady = false
                Log.e(TAG, "Text-to-speech initialization failed")
            }
        }
    }

    private fun startCamera() {
        val context = context ?: return

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener(
            {
                if (_fragmentCameraBinding == null ||
                    !isAdded ||
                    !viewLifecycleOwner.lifecycle.currentState.isAtLeast(
                        androidx.lifecycle.Lifecycle.State.STARTED
                    )
                ) {
                    return@addListener
                }

                try {
                    cameraProvider = cameraProviderFuture.get()
                    bindCameraUseCases()
                } catch (e: Exception) {
                    Log.e(TAG, "Camera initialization failed", e)
                    showError("Unable to initialize camera")
                }
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val binding = _fragmentCameraBinding ?: return

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(
                    binding.viewFinder.surfaceProvider
                )
            }

        imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(
                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            )
            .setOutputImageFormat(
                ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
            )
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzeImage(imageProxy)
                }
            }

        try {
            provider.unbindAll()

            provider.bindToLifecycle(
                viewLifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.d(TAG, "Camera successfully started")

        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
            showError("Unable to start camera")
        }
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy)

            val rotation =
                imageProxy.imageInfo.rotationDegrees

            objectDetectorHelper.detect(
                bitmap,
                rotation
            )

        } catch (e: Exception) {
            Log.e(TAG, "Image analysis failed", e)

        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(
        imageProxy: ImageProxy
    ): Bitmap {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer

        buffer.rewind()

        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val width = imageProxy.width
        val height = imageProxy.height

        val pixels = IntArray(width * height)

        val row = ByteArray(rowStride)

        for (y in 0 until height) {
            buffer.position(y * rowStride)
            buffer.get(row, 0, rowStride)

            for (x in 0 until width) {
                val offset = x * pixelStride

                val r = row[offset].toInt() and 0xFF
                val g = row[offset + 1].toInt() and 0xFF
                val b = row[offset + 2].toInt() and 0xFF
                val a = row[offset + 3].toInt() and 0xFF

                pixels[y * width + x] =
                    (a shl 24) or
                    (r shl 16) or
                    (g shl 8) or
                    b
            }
        }

        return Bitmap.createBitmap(
            pixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        val activity = activity ?: return

        activity.runOnUiThread {
            if (_fragmentCameraBinding == null) {
                return@runOnUiThread
            }

            processDetections(results)
        }
    }

    private fun processDetections(
        results: List<Detection>?
    ) {
        if (results.isNullOrEmpty()) {
            resetCandidate()
            return
        }

        val bestCategory = results
            .flatMap { it.categories }
            .filter {
                it.score >= SPEECH_CONFIDENCE_THRESHOLD
            }
            .maxByOrNull { it.score }

        if (bestCategory == null) {
            resetCandidate()
            return
        }

        val label = bestCategory.label
            .trim()
            .lowercase(Locale.US)

        if (label.isEmpty()) {
            resetCandidate()
            return
        }

        if (candidateLabel == label) {
            candidateCount++
        } else {
            candidateLabel = label
            candidateCount = 1
        }

        if (candidateCount < REQUIRED_STABLE_FRAMES) {
            return
        }

        val currentTime = SystemClock.elapsedRealtime()

        if (
            currentTime - lastAnnouncementTime <
            MIN_ANNOUNCEMENT_GAP_MS
        ) {
            return
        }

        if (
            label == lastAnnouncedLabel &&
            currentTime - lastLabelAnnouncementTime <
            SAME_LABEL_REPEAT_DELAY_MS
        ) {
            return
        }

        announceObject(label)

        lastAnnouncedLabel = label
        lastAnnouncementTime = currentTime
        lastLabelAnnouncementTime = currentTime

        resetCandidate()
    }

    private fun announceObject(label: String) {
        if (!ttsReady) {
            return
        }

        val speechText = when (label) {
            "person" -> "Person detected ahead"
            "car" -> "Car detected"
            "bicycle" -> "Bicycle detected"
            "motorcycle" -> "Motorcycle detected"
            "bus" -> "Bus detected"
            "truck" -> "Truck detected"
            "chair" -> "Chair detected"
            "dog" -> "Dog detected"
            "cat" -> "Cat detected"
            else -> "$label detected"
        }

        utteranceCounter++

        val utteranceId =
            "object_detection_$utteranceCounter"

        textToSpeech?.speak(
            speechText,
            TextToSpeech.QUEUE_FLUSH,
            null,
            utteranceId
        )

        Log.d(TAG, "Announced: $speechText")
    }

    private fun resetCandidate() {
        candidateLabel = ""
        candidateCount = 0
    }

    override fun onError(error: String) {
        Log.e(TAG, error)
        showError(error)
    }

    private fun showError(message: String) {
        activity?.runOnUiThread {
            if (isAdded && _fragmentCameraBinding != null) {
                Toast.makeText(
                    requireContext(),
                    message,
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    override fun onDestroyView() {
        cameraProvider?.unbindAll()
        cameraProvider = null

        imageAnalyzer?.clearAnalyzer()
        imageAnalyzer = null
        preview = null

        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }

        if (::objectDetectorHelper.isInitialized) {
            objectDetectorHelper.clearObjectDetector()
        }

        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false

        resetCandidate()

        _fragmentCameraBinding = null

        super.onDestroyView()
    }
}
