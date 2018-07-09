package com.github.arc11295.slitherlinkassistant;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        clearAppFiles();
        try {
            copyTesseractData();
        } catch (IOException e) {
            Log.e(TAG, "onCreate: something went wrong while copying tesseract data", e);
        }
        setContentView(R.layout.activity_main);
    }

    private void copyTesseractData() throws IOException {
        Log.d(TAG, "copyTesseractData() called");
        File filesdir = getFilesDir();
        ProcessImageActivity.sTessParent = filesdir.getPath();
        if (!ProcessImageActivity.sTessParent.endsWith("/")) {
            ProcessImageActivity.sTessParent += "/";
        }
        File tessdata = new File(filesdir, "/tessdata/");
        if (tessdata.mkdir()) {
            Log.d(TAG, "copyTesseractData: copying files");
            InputStream rawEng = getResources().openRawResource(R.raw.eng);
            File engFile = new File(tessdata, "/eng.traineddata");
            myWriteFile(engFile, rawEng);
            rawEng.close();
        }
    }

    private void clearAppFiles() {
        File filesdir = getFilesDir();
        File tessdata = new File(filesdir, "/tessdata/");
        if (tessdata.exists()) {
            File engFile = new File(tessdata, "/eng.traineddata");
            if (engFile.exists()) {
                engFile.delete();
            }
            tessdata.delete();
        }
    }

    private void myWriteFile(File dest, InputStream src) throws IOException{
        if (dest.createNewFile()) {
            FileOutputStream oStream = new FileOutputStream(dest);
            byte[] buffer = new byte[4096];
            while (src.read(buffer) != -1) {
                oStream.write(buffer);
            }
            oStream.close();
        }
    }

    public void openFullscreen(View view) {
        Intent intent = new Intent(this, FullscreenActivity.class);
        startActivity(intent);
        Log.d(TAG, "openFullscreen: successfully called startActivity and returned");
    }
}
