package org.a3dgc.terraingeneration;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.opengl.Matrix;

import org.a3dgc.terraingeneration.Geometry.Plane;

/**
 * This class implements our custom renderer. Note that the GL10 parameter passed in is unused for OpenGL ES 2.0
 * renderers -- the static class GLES20 is used instead.
 */
public class MainGLRenderer implements GLSurfaceView.Renderer {
    long start = (System.currentTimeMillis());
    private float mAngleX = 0f;
    private float mAngleY = 0f;
    private float xrot = 0f;
    private float yrot = 0f;
    private float a = 0f;
    private float b = 0f;
    float posX = 0;
    float posY = 0;
    float oldPosX = 0;
    float oldPosY = 0;
    private final float modFactor = 60f;
    /**
     * Store the model matrix. This matrix is used to move models from object space (where each model can be thought
     * of being located at the center of the universe) to world space.
     */
    private float[] mModelMatrix = new float[16];

    /**
     * Store the view matrix. This can be thought of as our camera. This matrix transforms world space to eye space;
     * it positions things relative to our eye.
     */
    private float[] mViewMatrix = new float[16];

    /**
     * Store the projection matrix. This is used to project the scene onto a 2D viewport.
     */
    private float[] mProjectionMatrix = new float[16];

    /**
     * Allocate storage for the final combined matrix. This will be passed into the shader program.
     */
    private float[] mMVPMatrix = new float[16];


    /**
     * Store the accumulated rotation.
     */
    private final float[] mAccumulatedRotation = new float[16];

    /**
     * Store the current rotation.
     */
    private final float[] mCurrentRotation = new float[16];

    private final float[] mTemporaryMatrix = new float[16];

    private final float[] mCurrentTranslation = new float[16];

    private final int vertexNumber = 100;
    private final int size = 40;

    /**
     * This will be used to pass in the transformation matrix.
     */
    private int mMVPMatrixHandle;

    /**
     * This will be used to pass in model position information.
     */
    private int mPositionHandle;

    /**
     * This will be used to pass in model color information.
     */
    private int mColorHandle;

    /**
     * This will be used to pass in the texture.
     */
    private int mTextureUniformHandle;

    /**
     * This is a handle to our texture data.
     */
    private int mTextureDataHandle;

    /**
     * How many bytes per float.
     */
    private final int mBytesPerFloat = 4;

    /**
     * How many elements per vertex.
     */
    private final int mStrideBytes = 7 * mBytesPerFloat;

    /**
     * Offset of the position data.
     */
    private final int mPositionOffset = 0;

    /**
     * Size of the position data in elements.
     */
    private final int mPositionDataSize = 3;

    /**
     * Offset of the color data.
     */
    private final int mColorOffset = 3;

    /**
     * Size of the color data in elements.
     */
    private final int mColorDataSize = 4;

    int programHandle;
    int mTimeHandle;

    FloatBuffer[] worldFB = new FloatBuffer[9];
    Plane[] world = new Plane[9];

    //light stuff
    private float[] mLightModelMatrix = new float[16];
    /**
     * This will be used to pass in the modelview matrix.
     */
    private int mMVMatrixHandle;

    /**
     * This will be used to pass in the light position.
     */
    private int mLightPosHandle;

    private final float[] mLightPosInModelSpace = new float[]{0.0f, 4.0f, 0.0f, 1.0f};

    /**
     * Used to hold the current position of the light in world space (after transformation via model matrix).
     */
    private final float[] mLightPosInWorldSpace = new float[4];

    /**
     * Used to hold the transformed position of the light in eye space (after transformation via modelview matrix)
     */
    private final float[] mLightPosInEyeSpace = new float[4];

    public MainGLRenderer() {
        int index = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                world[index] = new Plane(size, size, vertexNumber, vertexNumber, integerPart(yrot, size) + j, integerPart(xrot, size) + i);
                worldFB[index] = ByteBuffer.allocateDirect(world[index].vertex.length * mBytesPerFloat)
                        .order(ByteOrder.nativeOrder()).asFloatBuffer();
                worldFB[index].put(world[index].vertex).position(0);
                index++;
            }
        }
    }

    @Override
    public void onSurfaceCreated(GL10 glUnused, EGLConfig config) {
        // Set the background clear color to gray.
        GLES20.glClearColor(0.184314f, 0.184314f, 0.309804f, 0.5f);
        // Use culling to remove back faces.
        //GLES20.glEnable(GLES20.GL_CULL_FACE);

        // Enable depth testing
        //GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        // Position the eye behind the origin.
        final float eyeX = 0.0f + xrot;
        final float eyeY = 7f;
        final float eyeZ = 0f + yrot;

        // We are looking toward the distance
        final float lookX = 0.0f + xrot;
        final float lookY = 0.0f;
        final float lookZ = -5.0f + yrot;

        // Set our up vector. This is where our head would be pointing were we holding the camera.
        final float upX = 0.0f;
        final float upY = 1.0f;
        final float upZ = 0.0f;

        // Set the view matrix. This matrix can be said to represent the camera position.
        // NOTE: In OpenGL 1, a ModelView matrix is used, which is a combination of a model and
        // view matrix. In OpenGL 2, we can keep track of these matrices separately if we choose.
        Matrix.setIdentityM(mViewMatrix, 0);
        Matrix.setLookAtM(mViewMatrix, 0, eyeX, eyeY, eyeZ, lookX, lookY, lookZ, upX, upY, upZ);

        Matrix.setIdentityM(mAccumulatedRotation, 0);

        final String vertexShader =
                "uniform mat4 u_MVPMatrix;      \n"        // A constant representing the combined model/view/projection matrix.
                        + "uniform mat4 u_MVMatrix;      \n"        // A constant representing the combined model/view matrix.
                        + "uniform vec3 u_LightPos;       \n"        // The position of the light in eye space.
                        + "attribute vec4 a_Position;     \n"        // Per-vertex position information we will pass in.
                        + "attribute vec4 a_Color;        \n"        // Per-vertex color information we will pass in.
                        + "uniform float u_Time;\n"
                        + "varying vec4 v_Color;          \n"
                        + "varying vec4 v_Position;       \n"
                        + "varying float v_Time;       \n"
                        + "varying float v_diff;           \n"
                        + "varying vec4 v_Normal;         \n"
                        + "void main()                    \n"        // The entry point for our vertex shader.
                        + "{"
                        + "if(a_Position.z<-0.5){ \n"
                        // Transform the vertex into eye space.
                        + "   vec3 modelViewVertex = vec3(u_MVMatrix * a_Position);              \n"
                        // Transform the normal's orientation into eye space.
                        + "   vec3 modelViewNormal = vec3(u_MVMatrix * a_Color);     \n"
                        // Will be used for attenuation.
                        + "   float distance = length(u_LightPos - modelViewVertex);             \n"
                        // Get a lighting direction vector from the light to the vertex.
                        + "   vec3 lightVector = normalize(u_LightPos - modelViewVertex);        \n"
                        // Calculate the dot product of the light vector and vertex normal. If the normal and light vector are
                        // pointing in the same direction then it will get max illumination.
                        + "   float diffuse = max(dot(modelViewNormal, lightVector), 0.1);       \n"
                        // Attenuate the light based on distance.
                        + "   v_diff = 500.0*diffuse * (1.0 / (1.0 + (0.25 * distance * distance)));  \n"

                        + "   gl_Position = u_MVPMatrix* a_Position;"    // gl_Position is a special variable used to store the final position.
                        + "   v_Position = a_Position;          \n"        // Pass the color through to the fragment shader.
                        + "   v_Time = u_Time;        \n"
                        + "   v_Normal = a_Color;     }     \n"
                        + "if(a_Position.z>=-0.5){ \n"
                        + ""
                        + "   vec3 modelViewVertex = vec3(u_MVMatrix * a_Position);              \n"
                        // Transform the normal's orientation into eye space.
                        + "   vec3 modelViewNormal = vec3(u_MVMatrix * vec4(0.0,0.1,0.0,0.0));     \n"
                        // Will be used for attenuation.
                        + "   float distance = length(u_LightPos - modelViewVertex);             \n"
                        // Get a lighting direction vector from the light to the vertex.
                        + "   vec3 lightVector = normalize(u_LightPos - modelViewVertex);        \n"
                        // Calculate the dot product of the light vector and vertex normal. If the normal and light vector are
                        // pointing in the same direction then it will get max illumination.
                        + "   float diffuse = max(dot(modelViewNormal, lightVector), 0.1);       \n"
                        // Attenuate the light based on distance.
                        + "   v_diff = 500.0*diffuse * (1.0 / (1.0 + (0.25 * distance * distance)));  \n"
                        + "float frequency=2.0*3.14/0.1; float phase=0.01*frequency;"
                        + "float wave=a_Position.z+mod(abs(cos(a_Position.x*u_Time)/10.0),0.1)+mod(abs(cos(a_Position.y*u_Time)/10.0),0.1);float max=a_Position.z+0.2;"
                        + "vec4 sea=vec4(a_Position.x,a_Position.y,clamp(wave,a_Position.z,max),a_Position.w);\n"
                        + ""
                        + "   gl_Position = u_MVPMatrix   \n"
                        + "   *sea;   \n"
                        + "   v_Position = sea;   \n"
                        + "} }                              \n";    // normalized screen coordinates.

        final String fragmentShader =
                "uniform sampler2D u_Texture;    \n"
                        + "uniform sampler2D u_SeaTexture;"
                        + "precision highp float;        \n"
                        + "varying float v_Time;       \n"
                        + "varying vec4 v_Color;          \n"
                        + "varying float v_diff;          \n"
                        + "varying vec4 v_Position;       \n"
                        + "varying vec4 v_Normal;       \n"
                        + "void main()                    \n"
                        + "{"
                        + "vec4 n=normalize(v_Normal); \n"
                        + "if(v_Position.z<-0.5){ \n"
                        + "float pos=(v_Position[2]+0.5)/2.5;\n"
                        + "vec2 alt=vec2(n.x,pos);\n"
                        + "vec3 sample= texture2D(u_Texture,alt).xyz;     \n"
                        + "vec4 color=vec4(sample,1.0); \n"
                        + "gl_FragColor = color*v_diff; }"
                        + "if(v_Position.z>=-0.5){ \n"
                        + "float pos=(v_Position[2]+3.0)/2.5;\n"
                        + "vec2 altt=vec2(0.0,0.0);\n"
                        + "vec3 samplee= texture2D(u_SeaTexture,altt).xyz;     \n"
                        + "vec4 colorr=vec4(samplee.x,samplee.y,samplee.z,1.0); \n"
                        + "gl_FragColor = colorr*v_diff;"
                        + "} }                             \n";
        // Load in the vertex shader.
        int vertexShaderHandle = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);

        if (vertexShaderHandle != 0) {
            // Pass in the shader source.
            GLES20.glShaderSource(vertexShaderHandle, vertexShader);

            // Compile the shader.
            GLES20.glCompileShader(vertexShaderHandle);

            // Get the compilation status.
            final int[] compileStatus = new int[1];
            GLES20.glGetShaderiv(vertexShaderHandle, GLES20.GL_COMPILE_STATUS, compileStatus, 0);

            // If the compilation failed, delete the shader.
            if (compileStatus[0] == 0) {
                GLES20.glDeleteShader(vertexShaderHandle);
                vertexShaderHandle = 0;
            }
        }

        if (vertexShaderHandle == 0) {
            throw new RuntimeException("Error creating vertex shader.");
        }

        // Load in the fragment shader shader.
        int fragmentShaderHandle = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);

        if (fragmentShaderHandle != 0) {
            // Pass in the shader source.
            GLES20.glShaderSource(fragmentShaderHandle, fragmentShader);

            // Compile the shader.
            GLES20.glCompileShader(fragmentShaderHandle);

            // Get the compilation status.
            final int[] compileStatus = new int[1];
            GLES20.glGetShaderiv(fragmentShaderHandle, GLES20.GL_COMPILE_STATUS, compileStatus, 0);

            // If the compilation failed, delete the shader.
            if (compileStatus[0] == 0) {
                GLES20.glDeleteShader(fragmentShaderHandle);
                fragmentShaderHandle = 0;
            }
        }

        if (fragmentShaderHandle == 0) {
            throw new RuntimeException("Error creating fragment shader.");
        }

        // Create a program object and store the handle to it.
        programHandle = GLES20.glCreateProgram();

        if (programHandle != 0) {
            // Bind the vertex shader to the program.
            GLES20.glAttachShader(programHandle, vertexShaderHandle);

            // Bind the fragment shader to the program.
            GLES20.glAttachShader(programHandle, fragmentShaderHandle);

            // Bind attributes
            GLES20.glBindAttribLocation(programHandle, 0, "a_Position");
            GLES20.glBindAttribLocation(programHandle, 1, "a_Color");

            // Link the two shaders together into a program.
            GLES20.glLinkProgram(programHandle);

            // Get the link status.
            final int[] linkStatus = new int[1];
            GLES20.glGetProgramiv(programHandle, GLES20.GL_LINK_STATUS, linkStatus, 0);

            // If the link failed, delete the program.
            if (linkStatus[0] == 0) {
                GLES20.glDeleteProgram(programHandle);
                programHandle = 0;
            }
        }

        if (programHandle == 0) {
            throw new RuntimeException("Error creating program.");
        }

        // Set program handles. These will later be used to pass in values to the program.
        mMVPMatrixHandle = GLES20.glGetUniformLocation(programHandle, "u_MVPMatrix");
        mPositionHandle = GLES20.glGetAttribLocation(programHandle, "a_Position");
        mColorHandle = GLES20.glGetAttribLocation(programHandle, "a_Color");
        mTextureUniformHandle = GLES20.glGetUniformLocation(programHandle, "u_Texture");
        mTextureDataHandle = loadTexture(MainActivity.context, R.drawable.terrainnormal2, 0);
        int mSTextureUniformHandle = GLES20.glGetUniformLocation(programHandle, "u_SeaTexture");
        int mSTextureDataHandle = loadTexture(MainActivity.context, R.drawable.textureterrainsea, 1);
        float time = (System.currentTimeMillis() % modFactor) - start;
        mTimeHandle = GLES20.glGetUniformLocation(programHandle, "u_Time");


        // Tell OpenGL to use this program when rendering.
        GLES20.glUseProgram(programHandle);
        // Tell the texture uniform sampler to use this texture in the shader by binding to texture unit 0.
        GLES20.glUniform1i(mTextureUniformHandle, 0);
        GLES20.glUniform1i(mSTextureUniformHandle, 1);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mTextureDataHandle);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + 1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mSTextureDataHandle);


        Matrix.translateM(mModelMatrix, 0, 0.0f, -2.0f, 0.0f);
    }

    @Override
    public void onSurfaceChanged(GL10 glUnused, int width, int height) {

        // Set the OpenGL viewport to the same vertexNumber as the surface.
        GLES20.glViewport(0, 0, width, height);

        // Create a new perspective projection matrix. The height will stay the same
        // while the width will vary as per aspect ratio.
        final float ratio = (float) width / height;
        final float left = -ratio;
        final float right = ratio;
        final float bottom = -1.0f;
        final float top = 1.0f;
        final float near = 1.0f;
        final float far = 30.0f;

        Matrix.frustumM(mProjectionMatrix, 0, left, right, bottom, top, near, far);
    }

    @Override
    public void onDrawFrame(GL10 glUnused) {

        mLightPosHandle = GLES20.glGetUniformLocation(programHandle, "u_LightPos");
        mMVMatrixHandle = GLES20.glGetUniformLocation(programHandle, "u_MVMatrix");
        GLES20.glUseProgram(programHandle);

        Matrix.setIdentityM(mLightModelMatrix, 0);

        Matrix.translateM(mLightModelMatrix, 0, -xrot, 10.0f, -yrot);

        Matrix.multiplyMV(mLightPosInWorldSpace, 0, mLightModelMatrix, 0, mLightPosInModelSpace, 0);
        Matrix.multiplyMV(mLightPosInEyeSpace, 0, mViewMatrix, 0, mLightPosInWorldSpace, 0);

        posX = integerPart(xrot, size);
        posY = integerPart(yrot, size);

        if (posX != oldPosX || posY != oldPosY) {

            int index = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    world[index] = new Plane(size, size, vertexNumber, vertexNumber, -posY + j, -posX + i);
                    worldFB[index] = ByteBuffer.allocateDirect(world[index].vertex.length * mBytesPerFloat)
                            .order(ByteOrder.nativeOrder()).asFloatBuffer();
                    worldFB[index].put(world[index].vertex).position(0);
                    index++;
                }
            }
            oldPosX = posX;
            oldPosY = posY;
        }

        long time = (System.currentTimeMillis() - start) / 300l;
        GLES20.glUniform1f(mTimeHandle, time);
        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);

        for (int i = 0; i < world.length; i++) {
            Matrix.setIdentityM(mModelMatrix, 0);
            Matrix.rotateM(mModelMatrix, 0, 90.0f, 1.0f, 0.0f, 0.0f);
            drawPlane(worldFB[i], world[i]);
        }
        Matrix.setIdentityM(mCurrentTranslation, 0);
        Matrix.translateM(mCurrentTranslation, 0, xrot, -7.0f, yrot);

        Matrix.setIdentityM(mViewMatrix, 0);
        Matrix.setIdentityM(mCurrentRotation, 0);
        Matrix.rotateM(mCurrentRotation, 0, mAngleX, 0.0f, 1.0f, 0.0f);

        mAngleX = 0.0f;
        mAngleY = 0.0f;

        Matrix.multiplyMM(mTemporaryMatrix, 0, mCurrentRotation, 0, mAccumulatedRotation, 0);
        System.arraycopy(mTemporaryMatrix, 0, mAccumulatedRotation, 0, 16);


    }

    private void drawPlane(final FloatBuffer aPlaneBuffer, final Plane p) {
        aPlaneBuffer.position(mPositionOffset);
        GLES20.glVertexAttribPointer(mPositionHandle, mPositionDataSize, GLES20.GL_FLOAT, false,
                mStrideBytes, aPlaneBuffer);

        GLES20.glEnableVertexAttribArray(mPositionHandle);

        // Pass in the color information
        aPlaneBuffer.position(mColorOffset);
        GLES20.glVertexAttribPointer(mColorHandle, mColorDataSize, GLES20.GL_FLOAT, false,
                mStrideBytes, aPlaneBuffer);

        GLES20.glEnableVertexAttribArray(mColorHandle);

        // This multiplies the view matrix by the model matrix, and stores the result in the MVP matrix
        // (which currently contains model * view).

        Matrix.multiplyMM(mMVPMatrix, 0, mViewMatrix, 0, mModelMatrix, 0);
        GLES20.glUniformMatrix4fv(mMVMatrixHandle, 1, false, mMVPMatrix, 0);
        Matrix.multiplyMM(mMVPMatrix, 0, mCurrentTranslation, 0, mMVPMatrix, 0);
        Matrix.multiplyMM(mMVPMatrix, 0, mTemporaryMatrix, 0, mMVPMatrix, 0);
        // This multiplies the modelview matrix by the projection matrix, and stores the result in the MVP matrix
        // (which now contains model * view * projection).
        Matrix.multiplyMM(mMVPMatrix, 0, mProjectionMatrix, 0, mMVPMatrix, 0);
        GLES20.glUniformMatrix4fv(mMVPMatrixHandle, 1, false, mMVPMatrix, 0);

        GLES20.glUniform3f(mLightPosHandle, mLightPosInEyeSpace[0], mLightPosInEyeSpace[1], mLightPosInEyeSpace[2]);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, ((vertexNumber + 1) * (vertexNumber + 1)) * 6,
                GLES20.GL_UNSIGNED_SHORT, p.indicesBuffer);

    }

    public static int loadShader(int type, String shaderCode) {

        // create a vertex shader type (GLES20.GL_VERTEX_SHADER)
        // or a fragment shader type (GLES20.GL_FRAGMENT_SHADER)
        int shader = GLES20.glCreateShader(type);

        // add the source code to the shader and compile it
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);

        return shader;
    }

    void setAngle(float x, float y) {
        mAngleX += x;
        mAngleY += y;
        a += x;
        b += y;

    }

    float getX() {
        return xrot;
    }

    float getY() {
        return yrot;
    }

    void setXYrot(float x, float y) {

        xrot -= (y * Math.sin((a * Math.PI) / 180f)) + (x * Math.cos((a * Math.PI) / 180f));
        yrot += (y * Math.cos((a * Math.PI) / 180f)) - (x * Math.sin((a * Math.PI) / 180f));
    }

    void resetAngle() {
        mAngleX = 0;
        mAngleY = 0;
    }

    private static int loadTexture(final Context context, final int resourceId, int i) {
        final int[] textureHandle = new int[1];

        GLES20.glGenTextures(1, textureHandle, 0);

        if (textureHandle[0] != 0) {
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inScaled = false;   // No pre-scaling

            // Read in the resource
            final Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), resourceId, options);

            // Bind to the texture in OpenGL
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureHandle[0]);

            // Set filtering
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

            // Load the bitmap into the bound texture.
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);

            // Recycle the bitmap, since its data has been loaded into OpenGL.
            bitmap.recycle();
        }

        if (textureHandle[0] == 0) {
            throw new RuntimeException("Error loading texture.");
        }

        return textureHandle[0];
    }

    public static float integerPart(float n, float d) {
        return (n / d <= 0) ? (float) Math.ceil((n - 10f) / d) + 0.0f : (float) Math.floor((n + 10f) / d) + 0.0f;
    }

}
