package com.example.ckd_predict_anzules_midterm

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.res.AssetFileDescriptor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

import kotlin.math.pow

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements
        val inputBp: EditText = findViewById(R.id.input_bp)
        val inputBu: EditText = findViewById(R.id.input_bu)
        val inputSc: EditText = findViewById(R.id.input_sc)
        val buttonPredict: Button = findViewById(R.id.button_predict)
        val textResult: TextView = findViewById(R.id.text_result)

        // Load TFLite model
        interpreter = Interpreter(loadModelFile("ckd_logistic_regression_model_final.tflite"))

        buttonPredict.setOnClickListener {
            // Get input values
            val bp = inputBp.text.toString().toFloat()
            val bu = inputBu.text.toString().toFloat()
            val sc = inputSc.text.toString().toFloat()

            // Standardize input values
            val standardizedInput = floatArrayOf(
                (bp - 76.45945946f) / 13.81948212f,
                (bu - 57.87891892f) / 50.51830269f,
                (sc - 3.05959459f) / 5.75178197f
            )

            // Prepare input for model
            val inputBuffer = ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder()) // Allocate space for 3 floats
            inputBuffer.rewind()  // Reset buffer to the beginning

            // Fill the ByteBuffer with the standardized input values
            for (value in standardizedInput) {
                inputBuffer.putFloat(value)
            }

            // Make prediction
            val output = Array(1) { FloatArray(1) }
            interpreter.run(inputBuffer, output)

            // Display result
            val prediction = output[0][0]
            if (prediction > 0.5) {
                textResult.text = "Most likely CKD"
            } else {
                textResult.text = "Healthy"
            }
        }
    }

    // Load the TFLite model file from assets
    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        interpreter.close() // Close the interpreter when the activity is destroyed
        super.onDestroy()
    }
}
