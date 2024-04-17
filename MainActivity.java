package com.example.demo;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;

import com.example.demo.databinding.ActivityMainBinding;
import com.example.demo.ml.Model;
import com.example.demo.ml.Model3;
import com.example.demo.ml.Model32;
import com.example.demo.ml.Model4;

import androidx.activity.result.ActivityResultLauncher;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.view.View;

import androidx.core.content.FileProvider;

import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button selection, prediction, capture;
    TextView result;
    Bitmap bitmap;
    ImageView imageview;
    int imageSize=224;
    private static final int CAMERA_REQUEST = 1888;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //permission
        //getPermission();
        String[] labels= new String[2];
        int cnt = 0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line=bufferedReader.readLine();
            while (line!=null){
                labels[cnt] = line;
                cnt++;
                line=bufferedReader.readLine();
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        selection = findViewById(R.id.selectbutton);
        prediction = findViewById(R.id.prediction);
        capture = findViewById(R.id.capturebutton);
        result = findViewById(R.id.result);
        imageview = findViewById(R.id.ImageView);
        selection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);


            }
        });
        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
                }


        });
        prediction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                classifyImage(bitmap);



            }
        });

    }
    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (cameraIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(cameraIntent, CAMERA_REQUEST);
        } else {
            Toast.makeText(getApplicationContext(), "No camera app found", Toast.LENGTH_SHORT).show();
        }
    }



     public void classifyImage(Bitmap image){
       try {
           // Load the face anti-spoofing model
           //Model3 antiSpoofingModel = Model3.newInstance(getApplicationContext());
           Model32 antiSpoofingModel = Model32.newInstance(getApplicationContext());

           // Create inputs for the anti-spoofing model
           TensorBuffer inputTensor = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
           ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
           byteBuffer.order(ByteOrder.nativeOrder());

           int[] intValues = new int[imageSize * imageSize];
           image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
           int pixel = 0;
           for (int i = 0; i < imageSize; i++) {
               for (int j = 0; j < imageSize; j++) {
                   int val = intValues[pixel++];
                   byteBuffer.putFloat(((val >> 16) & 0xFF) * 1.f / 1);
                   byteBuffer.putFloat(((val >> 8) & 0xFF) * 1.f / 1);
                   byteBuffer.putFloat((val & 0xFF) * 1.f / 1);
               }
           }

           // Load the image data into the input tensor
           inputTensor.loadBuffer(byteBuffer);

           // Run inference on the anti-spoofing model
           //Model3.Outputs outputs = antiSpoofingModel.process(inputTensor);
           Model32.Outputs outputs = antiSpoofingModel.process(inputTensor);
           TensorBuffer outputTensor = outputs.getOutputFeature0AsTensorBuffer();

           // Assuming the output tensor contains a single confidence score,
           // where higher values indicate a real face and lower values indicate a spoofed face
           float confidenceScore = outputTensor.getFloatArray()[0];

           // Define a threshold to classify real vs spoofed faces
           float threshold = 0.5f; // Adjust this threshold as needed

           // Check if the confidence score exceeds the threshold
           if (confidenceScore >= threshold) {
               // Real face detected
               result.setText("Real");
           } else {
               // Spoofed face detected
               result.setText("Spoof");
           }

           // Release model resources
           antiSpoofingModel.close();
       } catch (IOException e) {
           // Handle IOException
       }

   }

   /* public void classifyImage(Bitmap image){
        try {
            // Load the face anti-spoofing model
            //Model3 antiSpoofingModel = Model3.newInstance(getApplicationContext());
            Model4 antiSpoofingModel = Model4.newInstance(getApplicationContext());

            // Create inputs for the anti-spoofing model
            TensorBuffer inputTensor = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * 1.f / 1);
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * 1.f / 1);
                    byteBuffer.putFloat((val & 0xFF) * 1.f / 1);
                }
            }

            // Load the image data into the input tensor
            inputTensor.loadBuffer(byteBuffer);

            // Run inference on the anti-spoofing model
            //Model3.Outputs outputs = antiSpoofingModel.process(inputTensor);
            Model4.Outputs outputs = antiSpoofingModel.process(inputTensor);
            TensorBuffer outputTensor = outputs.getOutputFeature0AsTensorBuffer();

            // Assuming the output tensor contains a single confidence score,
            // where higher values indicate a real face and lower values indicate a spoofed face
            float confidenceScore = outputTensor.getFloatArray()[0];

            // Define a threshold to classify real vs spoofed faces
            float threshold = 0.5f; // Adjust this threshold as needed

            // Check if the confidence score exceeds the threshold
            if (confidenceScore >= threshold) {
                // Real face detected
                result.setText("Real");
            } else {
                // Spoofed face detected
                result.setText("Spoof");
            }

            // Release model resources
            antiSpoofingModel.close();
        } catch (IOException e) {
            // Handle IOException
        }

    }*/
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==11){
            if(grantResults.length>0){
                if(grantResults[0]==PackageManager.PERMISSION_GRANTED){
                    //this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==10){
            if(data!=null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);


                    int dimension = Math.min(bitmap.getWidth(),bitmap.getHeight());
                    bitmap = ThumbnailUtils.extractThumbnail(bitmap,dimension,dimension);
                    imageview.setImageBitmap(bitmap);
                    bitmap = Bitmap.createScaledBitmap(bitmap,imageSize,imageSize,false);

                } catch (IOException e) {
                    e.printStackTrace();
                    ;
                }
            }
        }
        else if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && data != null) {
            Bundle extras = data.getExtras();
            if (extras != null) {
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                if (imageBitmap != null) {
                    // Resize and set bitmap to imageView
                    int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
                    imageBitmap = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension);
                    imageview.setImageBitmap(imageBitmap);

                    // Resize bitmap to desired size
                    bitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false);
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
