package com.github.arc11295.slitherlinkassistant;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void openFullscreen(View view) {
        Intent intent = new Intent(this, FullscreenActivity.class);
        startActivity(intent);
        Log.d(TAG, "openFullscreen: successfully called startActivity and returned");
    }
}
