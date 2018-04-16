package com.github.arc11295.slitherlinkassistant;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.drawable.ShapeDrawable;
import android.graphics.drawable.shapes.RectShape;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

/**
 * TODO: document your custom view class.
 * TODO: make various attributes styleable (e.g. the color of the outline for the crop area)
 */
public class CropBoxView extends View {
    private static final String TAG = "CropBoxView";
    private ShapeDrawable mFrame;
    private ShapeDrawable mInnerBox;
    private ShapeDrawable mBoxOutline;
    private Bitmap mBitmap;
    private Paint mPaint;
    private Rect mCropArea;

    private static final int FRAME_ALPHA = 150;
    private static final double FRAME_MARGIN = 0.05;

    public CropBoxView(Context context, AttributeSet attrs) {
        super(context, attrs);

        final TypedArray a = getContext().obtainStyledAttributes(
                attrs, R.styleable.CropBoxView, 0, 0);


        mFrame = new ShapeDrawable(new RectShape());
        mFrame.getPaint().setColor(Color.GRAY);
        mFrame.setAlpha(FRAME_ALPHA);

        mInnerBox = new ShapeDrawable(new RectShape());
        Paint boxPaint = mInnerBox.getPaint();
        boxPaint.setAlpha(0);
        boxPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));

        mBoxOutline = new ShapeDrawable(new RectShape());
        Paint outlinePaint = mBoxOutline.getPaint();
        outlinePaint.setColor(Color.RED);
        outlinePaint.setStyle(Paint.Style.STROKE);

        mPaint = new Paint();

        a.recycle();
    }

    public Rect getCropArea() {
        return new Rect(mCropArea);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        Log.d(TAG, "onSizeChanged: width is "+w);
        Log.d(TAG, "onSizeChanged: height is "+h);

        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(mBitmap);

        mFrame.setBounds(0, 0, w, h);
        int left = (int) (FRAME_MARGIN * w);
        int boxSize = (int) (((double) w) - (2*FRAME_MARGIN*w));
        int right = left + boxSize;
        int top = (h - boxSize)/2;
        int bot = (h + boxSize)/2;
        mCropArea = new Rect(left, top, right, bot);
        mInnerBox.setBounds(mCropArea);
        mBoxOutline.setBounds(mCropArea);
        mFrame.draw(canvas);
        mInnerBox.draw(canvas);
        mBoxOutline.draw(canvas);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        canvas.drawBitmap(mBitmap, 0 , 0, mPaint);
    }
}
