#ifdef GL_ES
precision mediump float;
#endif

uniform float time;
uniform vec2 resolution;

#define iTime time
#define iResolution resolution

uniform float     iChannelTime[4];
uniform float     iBlockOffset; 
uniform vec4      iDate;
uniform float     iSampleRate;
uniform vec3      iChannelResolution[4];
uniform sampler2D iChannel0;

vec2 mainSound( float time )
{
  time = mod( time, 40.0 );

  // do 3 echo/reverb bounces
  vec2 tot = vec2(0.0);
  for( int i=0; i<3; i++ )
  {
    float h = float(i)/(3.0-1.0);

    // compute note 
    float t = (time - 0.53*h)/0.18;
    float n = 0.0, b = 0.0, x = 0.0;
    #define D(u,v)   b+=float(u);if(t>b){x=b;n=float(v);}
    D(10,71)D(2,76)D(3,79)D(1,78)D( 2,76)D( 4,83)D(2,81)D(6,78)D(6,76)D(3,79)
    D( 1,78)D(2,74)D(4,77)D(2,71)D(10,71)D( 2,76)D(3,79)D(1,78)D(2,76)D(4,83)
    D( 2,86)D(4,85)D(2,84)D(4,80)D( 2,84)D( 3,83)D(1,82)D(2,71)D(4,79)D(2,76)
    D(10,79)D(2,83)D(4,79)D(2,83)D( 4,79)D( 2,84)D(4,83)D(2,82)D(4,78)D(2,79)
    D( 3,83)D(1,82)D(2,70)D(4,71)D( 2,83)D(10,79)D(2,83)D(4,79)D(2,83)D(4,79)
    D( 2,86)D(4,85)D(2,84)D(4,80)D( 2,84)D( 3,83)D(1,82)D(2,71)D(4,79)D(2,76) 
        
    // calc frequency and time for note   
    float noteFreq = 440.0*pow( 2.0, (n-69.0)/12.0 );
    float noteTime = 0.18*(t-x);
    
    // compute instrument   
    float y  = 0.5*sin(6.2831*1.00*noteFreq*noteTime)*exp(-0.0015*1.0*noteFreq*noteTime);
          y += 0.3*sin(6.2831*2.01*noteFreq*noteTime)*exp(-0.0015*2.0*noteFreq*noteTime);
          y += 0.2*sin(6.2831*4.01*noteFreq*noteTime)*exp(-0.0015*4.0*noteFreq*noteTime);
          y += 0.1*y*y*y;     
          y *= 0.9 + 0.1*cos(40.0*noteTime);
          y *= smoothstep(0.0,0.01,noteTime);
          
    // accumulate echo    
    tot += y * vec2(0.5+0.2*h,0.5-0.2*h) * (1.0-sqrt(h)*0.85);
  }
  tot /= 3.0;
    
  return tot;
}


// --------[ Original Shadertoy ends here ]---------- //

void main() {
   //mainImage(fragOutput, gl_FragCoord.xy);
   // compute time `t` based on the pixel we're about to write
   // the 512.0 means the texture is 512 pixels across so it's
   // using a 2 dimensional texture, 512 samples per row
   float t = iBlockOffset + ((gl_FragCoord.x-0.5) + (gl_FragCoord.y-0.5)*512.0)/iSampleRate;

   // Get the 2 values for left and right channels
   vec2 y = mainSound( t );

   // convert them from -1 to 1 to 0 to 65536
   vec2 v  = floor((0.5+0.5*y)*65536.0);

   // separate them into low and high bytes
   vec2 vl = mod(v,256.0)/255.0;
   vec2 vh = floor(v/256.0)/255.0;

   // write them out where 
   // RED   = channel 0 low byte
   // GREEN = channel 0 high byte
   // BLUE  = channel 1 low byte
   // ALPHA = channel 2 high byte
   gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
}