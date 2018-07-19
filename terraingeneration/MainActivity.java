package org.a3dgc.terraingeneration;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.support.constraint.ConstraintLayout;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
public static boolean BUTTON=false;
    public static TextView tx;
    public static TextView ty;
    public static TextView px;
    public static TextView py;


    private GLSurfaceView mGLView;
    public static Context context;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //Request the absence of useless layout
        context=getApplicationContext();
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        // Create a GLSurfaceView instance and set it
        // as the ContentView for this Activity.


        mGLView = new MainGLSurfaceView(this);
        setContentView(mGLView);
        tx= new TextView(this);
        ty= new TextView(this);
        px= new TextView(this);
        py= new TextView(this);

        setView(tx,"PosX: 0",400f,20f);
        setView(ty,"PosY: 0",400f,80f);
        setView(px,"IndexX: 0",800f,20f);
        setView(py,"IndexY: 0",800f,80f);


        final Button b = new Button(this);

        b.setText("Rotate");
        b.setTag(1);

        b.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                BUTTON = !BUTTON;
                final int status =(Integer) v.getTag();
                if(status == 1) {
                    b.setText("Move");
                    v.setTag(0); //pause
                } else {
                    b.setText("Rotate");
                    v.setTag(1); //pause
                }
            }
        });

        this.addContentView(b,
                new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));
        this.addContentView(tx,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));
        this.addContentView(ty,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));
        this.addContentView(px,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));
        this.addContentView(py,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));

    }



    @Override
    protected void onResume()
    {
        // The activity must call the
        // GL surface view's onResume() on activity onResume().
        super.onResume();
        mGLView.onResume();
    }

    @Override
    protected void onPause()
    {
        // The activity must call the GL surface view's onPause() on activity onPause().
        super.onPause();
        mGLView.onPause();
    }
    public static void setTextX(String s){
        tx.setText(s);
    }
    public static void setTextY(String s){
        ty.setText(s);
    }public static void setTextIndexY(String s){
        py.setText(s);
    }public static void setTextIndexX(String s){
        px.setText(s);
    }
    private static void setView(TextView t,String s,float x, float y){

        t.setText(s);

        t.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,LinearLayout.LayoutParams.WRAP_CONTENT));

        t.setX(x);
        t.setY(y);


    }
}
