package org.a3dgc.terraingeneration.Geometry;

import org.a3dgc.terraingeneration.Utils.HeightMap;
/**
 * Created by Gianmarco.
 */
public class Plane extends Mesh {

    public float[] vertex;
    public float[] index;
    private static final int COORDS_PER_VERTEX = 7;
private int offset;
    private float offsetX;
    private float offsetY;
    private int vertexCount;
    private int vertexStride = COORDS_PER_VERTEX * 4; // 4 bytes per vertex


    public Plane(float width, float height, int widthSegments,
                 int heightSegments,float offsetY,float offsetX) {

        offset=widthSegments;
        HeightMap hp = new HeightMap(widthSegments);
        float[] vertices = new float[((widthSegments + 1) * (heightSegments + 1)) * (4 + 3)];

        short[] indices = new short[(widthSegments + 1) * (heightSegments + 1)
                * 6];
        vertexCount = vertices.length / COORDS_PER_VERTEX;
        this.offsetY=offsetY;
        this.offsetX=offsetX;
        float xOffset = width / -2;
        float yOffset = height / -2;
        float xWidth = width / (widthSegments);
        float yHeight = height / (heightSegments);
        int currentVertex = 0;
        int currentIndex = 0;
        float fromCenterX=offsetX*width;
        float fromCenterY=offsetY*height;
        float factor=offsetX/offsetX;
        double z0;
        double w0;
        short w = (short) (widthSegments + 1);
        for (int y = 0; y < heightSegments + 1; y++) {
            for (int x = 0; x < widthSegments + 1; x++) {
                vertices[currentVertex] = xOffset + x * xWidth +fromCenterX;
                vertices[currentVertex + 1] = yOffset + y * yHeight+fromCenterY;
                z0=HeightMap.noise((double)vertices[currentVertex],(double)vertices[currentVertex + 1]);
                //w0=HeightMap.noise((double)vertices[currentVertex],(double)vertices[currentVertex + 1],z0);
                vertices[currentVertex + 2] =sumOctave(10,(double)vertices[currentVertex],(double)vertices[currentVertex + 1],z0,
                        2f,0.5f,0.1f,-3,3);

                currentVertex += 7;


                int n = y * (widthSegments + 1) + x;

                if (y < heightSegments && x < widthSegments) {
                    // Face one
                    indices[currentIndex] = (short) n;
                    indices[currentIndex + 1] = (short) (n + 1);
                    indices[currentIndex + 2] = (short) (n + w);
                    // Face two
                    indices[currentIndex + 3] = (short) (n + 1);
                    indices[currentIndex + 4] = (short) (n + 1 + w);
                    indices[currentIndex + 5] = (short) (n + 1 + w - 1);

                    currentIndex += 6;
                }
            }
        }
        currentVertex=0;
        for (int y = 0; y <=heightSegments; y++) {
            for (int x = 0; x <= widthSegments; x++) {
                float hL = height(x-1,y,vertices);
                float hR = height(x+1,y,vertices);
                float hD = height(x,y-1,vertices);
                float hU = height(x,y+1,vertices);
                float magnitude=(float)Math.sqrt((hL-hR)*(hL-hR)+(hD-hU)*(hD-hU)+2*2);
                vertices[currentVertex + 3]=(hL-hR)/magnitude;
                vertices[currentVertex + 4]=(hD-hU)/magnitude;
                vertices[currentVertex + 5]=2f/magnitude;
                vertices[currentVertex + 6]=0.0f;
                currentVertex += 7;
            }
            }
        vertex = vertices;
        setIndices(indices);
        //setVertices(vertices);

    }

    public float sumOctave(int num_iterations,double  x,double  y,double z,float amplitude, float persistence, float scale, float low, float high){
        float maxAmp = 0;
        float  amp = amplitude;
        float freq = scale;
        float noise = 0;
        for(int i = 0; i < num_iterations; ++i) {
            noise +=(float) HeightMap.noise(x * freq, y * freq,z * freq) * amp;
            maxAmp += amp;
            amp *= persistence;
            freq *= 2;
        }
        noise /=maxAmp;
        noise = noise * (high - low) / 2 + (high + low) / 2;
        if(noise>-0.5f){
            noise=-0.5f;
        }
        return noise;
    }
    public float height(int x, int y,float array[]) {
        if(x<0){
            x=0;
        }
        if (x>offset){
            x=offset;
        }
        if(y<0){
            y=0;
        }
        if (y>offset){
            y=offset;
        }

        int index=(x+y*(offset+1))*7+2;
        return array[index];
    }
}