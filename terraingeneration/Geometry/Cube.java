package org.a3dgc.terraingeneration.Geometry;

import android.opengl.GLES30;

import org.a3dgc.terraingeneration.MainGLRenderer;

import javax.microedition.khronos.opengles.GL10;

/**
 * Created by Gianmarco.
 */

public class Cube extends Mesh {
public float[] vertices;
    static final int COORDS_PER_VERTEX = 3;
    private int mMVPMatrixHandle;
    private final int mProgram;

    float color[] = { 0.55f, 0.55f, 0.1f, 1.0f };

    private int mPositionHandle;
    private int mColorHandle;

    private int vertexCount;
    private int vertexStride = COORDS_PER_VERTEX * 4; // 4 bytes per vertex
    private final String vertexShaderCode =
            "attribute vec4 vPosition;" +
                   " uniform mat4 uMVPMatrix;"+
                    "void main() {" +
                    "  gl_Position = uMVPMatrix*vPosition;" +
                    "}";

    private final String fragmentShaderCode =
            "precision mediump float;" +
                    "uniform vec4 vColor;" +
                    "void main() {" +
                    "  gl_FragColor = vColor;" +
                    "}";
    public Cube(float width, float height, float depth) {


        width /= 2;
        height /= 2;
        depth /= 2;

        float vertices[] = { -width, -height, -depth, // 0
                width, -height, -depth, // 1
                width, height, -depth, // 2
                -width, height, -depth, // 3
                -width, -height, depth, // 4
                width, -height, depth, // 5
                width, height, depth, // 6
                -width, height, depth, // 7
        };
        vertexCount=vertices.length / COORDS_PER_VERTEX;
        short indices[] = { 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
                3, 3, 7, 4, 3, 4, 0, 4, 7, 6, 4, 6, 5, 3, 0, 1, 3, 1, 2, };

        setIndices(indices);
        setVertices(vertices);

        int vertexShader = MainGLRenderer.loadShader(GLES30.GL_VERTEX_SHADER,
                vertexShaderCode);
        int fragmentShader = MainGLRenderer.loadShader(GLES30.GL_FRAGMENT_SHADER,
                fragmentShaderCode);

        // create empty OpenGL ES Program
        mProgram = GLES30.glCreateProgram();

        // add the vertex shader to program
        GLES30.glAttachShader(mProgram, vertexShader);

        // add the fragment shader to program
        GLES30.glAttachShader(mProgram, fragmentShader);

        // creates OpenGL ES program executables
        GLES30.glLinkProgram(mProgram);
    }
    public void draw(float[] mvpMatrix) {
        // Add program to OpenGL ES environment
        GLES30.glUseProgram(mProgram);

        // get handle to vertex shader's vPosition member
        mPositionHandle = GLES30.glGetAttribLocation(mProgram, "vPosition");

        // Enable a handle to the triangle vertices
        GLES30.glEnableVertexAttribArray(mPositionHandle);

        // Prepare the triangle coordinate data
        GLES30.glVertexAttribPointer(mPositionHandle, COORDS_PER_VERTEX,
                GLES30.GL_FLOAT, false,
                vertexStride, verticesBuffer);

        // get handle to fragment shader's vColor member
        mColorHandle = GLES30.glGetUniformLocation(mProgram, "vColor");
        setColor(0.5f,0.5554f,0.1f,1f);
        // Set color for drawing the triangle
        GLES30.glUniform4fv(mColorHandle, 1, color, 0);
        mMVPMatrixHandle = GLES30.glGetUniformLocation(mProgram, "uMVPMatrix");

        // Pass the projection and view transformation to the shader
        GLES30.glUniformMatrix4fv(mMVPMatrixHandle, 1, false, mvpMatrix, 0);
        // Draw the triangle
        GLES30.glDrawElements(GL10.GL_TRIANGLES, numOfIndices,
                GL10.GL_UNSIGNED_SHORT, indicesBuffer);

        // Disable vertex array
        GLES30.glDisableVertexAttribArray(mPositionHandle);
    }


}