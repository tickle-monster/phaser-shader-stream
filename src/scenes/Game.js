var AudioShader = require('audio-shader');
var Speaker = require('audio-speaker');
var glslify = require('glslify');

import { Scene } from 'phaser';

export class Game extends Scene
{
    constructor ()
    {
        super('Game');
    }

    create ()
    {
    AudioShader(glslify(`// Notations :
// t = absolute time in seconds
// tb = time in beats
// te = time in seconds since start of envelope

#define TAU (2.*3.1415926)

// ESin --- Exponentially-decaying sine wave
//     f: frequency
//     d: decay rate
#define ESin(f,d) sin(TAU*(f)*t)*exp(-d*t)

#define midiratio(x) exp2((x)/12.)
#define midicps(x) (440.*midiratio((x)-69.))

const float bpm = 100.;
const float bps = bpm/60.; // beats per second
const float beatdur = 1./bps; // beat duration


float rand(float p)
{
    // Hash function by Dave Hoskins
    // https://www.shadertoy.com/view/4djSRW
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float noise(float s){
    // Noise is sampled at every integer s
    // If s = t*f, the resulting signal is close to a white noise
    // with a sharp cutoff at frequency f.
    
    // For some reason float(int(x)+1) is sometimes not the same as floor(x)+1.,
    // and the former produces fewer artifacts?
    int si = int(floor(s));
    float sf = fract(s);
    sf = sf*sf*(3.-2.*sf); // smoothstep(0,1,sf)
    return mix(rand(float(si)), rand(float(si+1)), sf) * 2. - 1.;
}

float coloredNoise(float t, float fc, float df)
{
    // Noise peak centered around frequency fc
    // containing frequencies between fc-df and fc+df
    
    // Assumes fc is an integer, to avoid problems with sin(large number).
    
    // Modulate df-wide noise by an fc-frequency sinusoid
    return sin(TAU*fract(fc*t))*noise(t*df);
}

vec2 coloredNoise2(float t, float fc, float df)
{
    // Noise peak centered around frequency fc
    // containing frequencies between fc-df and fc+df
    
    // Assumes fc is an integer, to avoid problems with sin(large number).
    
    // Modulate df-wide noise by an fc-frequency sinusoid
    return sin(TAU*fract(fc*t))*vec2(noise(t*df),noise(-1000.-t*df));
}

float sweep(float t, float dur)
{
    // Exponential sweep from 20kHz to 20Hz in "dur" seconds
    // Running freq: 20000*exp(-t/dt)
    // freq=20 at t=dur  ==>  20 = 20000*exp(-dur/dt)
    //                   ==>  exp(-dur/dt) = 1/1000
    //                   ==>  -dur/dt = log(1/1000)
    //                   ==>  dt = dur/log(1000)
    float dt = dur/log(1e3);
    float intfreq = 20000.*exp(-t/dt)*dt;
    float phase = TAU*fract(intfreq);
    float sig = sin(phase);
    float env = step(0., t) * smoothstep(dur, 0.7*dur, t);
    return sig*env*0.1;
}


float triBipolar(float x)
{
    // Triangle wave going from -1 to +1, starting at zero, 4-periodic.
    return 1. - abs(2.-mod(x+1.,4.));
}


float triUnipolar(float x)
{
    // Triangle wave going from 0 to +1, starting at zero, 2-periodic.
    return abs(1.-mod(x+1.,2.));
}

float fold(float x, float a, float b)
{
    // Force x to lie between a and b, "bounce back" when going too far.
    return triUnipolar((x-a)/(b-a)) * (b-a) + a;
}


float curve(float x, float a, float b, float curvature)
{
    // When x is on the a-side, outputs 0
    // When x is on the b-side, outputs 1
    // When x is between a and b, interpolates.
    // curvature = 0 -> linear interpolation
    // Negative curvature gives lower values, positive gives higher values.
    x = (x-a)/(b-a);
    x = clamp(x, 0., 1.);
    return pow(x, exp(-curvature));
}

float percEnv(float t, float atk, float rel, float cur)
{
    return curve(t, 0., atk, cur) * curve(t, atk+rel, atk, cur);
}

float rampIntegral(float x)
{
    // Integral of clamp(x,0.,1.).
    // Used to calculate phase with portamento
    return (x<=0.) ? 0. : (x < 1.) ? x*x*0.5 : x-0.5;
}

float phasePortamento(float t, float f1, float f2, float t1, float t2)
{
    // Integral of frequency of a note with portamento
    // Transition lasts from t1 to t2, with initial frequency f1 and final frequency f2.
    // freq(t) = f1 for t < t1
    // ... = f2 for t > t2
    // ... = linear interpolation in between
    
    
    // Change of variable : x = (t-t1)/(t2-t1).
    // dt = (t2-t1)*dx
    // int freq(t) dt = int freq(t(x)) * (t2-t1) dx
    // ... = int (f1 + saturate(x)*(f2-f1)) * (t2-t1) dx
    // ... = f1*t + rampIntegral(x)*(t2-t1).
    
    float phase = rampIntegral((t-t1)/(t2-t1)) * (f2-f1) * (t2-t1) + f1*t;
    phase = mod(phase, 1.);
    return phase;
}

///////////////////////////////////
////////////// SOUNDS /////////////
///////////////////////////////////


float lpfSaw3(float t, float f, float fc, float Q)
{
    // Low-pass-filtered sawtooth wave
    // arguments are time, frequency, cutoff frequency, and resonance quality factor
    // https://www.shadertoy.com/view/wsGyWm
    float omega_c = 2.*3.14159*fc/f; // relative
    t = f*t - floor(f*t);
    // Compute the exact response of a second order system with those parameters
    // (value and derivative are continuous)
    // It is expressed as
    // 1 - 2t + A exp(-omega_c*t/Q) * cos(omega_c*t+phi)
    // We need to compute the amplitude A and phase phi.
    float alpha = omega_c/Q, beta=exp(-alpha), c = cos(omega_c), s = sin(omega_c);
    float tanphi = (alpha*beta*c + beta*omega_c*s - alpha) / (omega_c + alpha*beta*s - beta*omega_c*c);
    // We could use more trigonometric identities to avoid computing the arctangent, but whatever.
    float phi = atan(tanphi);
    float A = -2./(cos(phi) - beta*cos(omega_c+phi));
    
    float v = 1.-2.*t + A*exp(-alpha*t) * cos(omega_c*t+phi);
    return v;
}


//////////////////////////////////////
////////////// INSTRUMENTS ///////////
//////////////////////////////////////

vec2 bassStab(float t, float te, float f)
{
    // "Analog bass" sound based on a filtered sawtooth wave
    float fc = 100. + 18000.*exp(-53.*te);
    float env = smoothstep(0.,0.001,te) * exp(-6.*te);
    vec2 sig = vec2(0);
    sig += vec2(1.) * lpfSaw3(t, f, fc, 1.);
    sig += vec2(-1,1) * 0.2 * lpfSaw3(t - 0.05, f+1., fc, 1.); // Widen in stereo
    return sig * env * 0.08;
}

float kick(float t, float te, float atk)
{
    float f0 = midicps(34.), df = 300., spd = 40.;
    // Instantaneous frequency
    float ifreq = f0 + df*exp(-te*spd);
    // Integrate to obtain
    float phase = TAU*f0*te + TAU*df/spd * (1.-exp(-te*spd));
    float env = exp(-8.*te) + 2.5*exp(-100.*te);
    env *= smoothstep(-1e-6,0.,te);
    env *= (1. + 0.3*smoothstep(0.01,0.0,te)); // Enhance attack
    float v = sin(phase) * env;
    float click = noise(20000.*te) * smoothstep(0.,0.001,te) * smoothstep(0.01,0.001,te) * 0.8;
    click += sweep(te, 0.005) * 2.;
    v /= 1. + 0.3*abs(v);
    v *= (1. + 4.*smoothstep(0.05,0.15,te)*smoothstep(0.2,0.1,te)); // Enhance tail
    v += click; // Don't distort the click
    return v * smoothstep(-1e-6,atk,te);
}

float snare(float t, float te, float atk)
{
    // Snare is "body" + "white noise" + "click"
    float wnoise = noise(20000.*te) + coloredNoise(te, 6500.,1000.)*0.3;
    float nenv = (exp(-5.*te) + exp(-30.*te)) * smoothstep(0.2,0.0,te) * 0.5;
    
    float spd = 50.;
    float phase = TAU*180.*te + TAU*80./spd * (1.-exp(-te*spd));
    float body = sin(phase) * 1.5 * smoothstep(0.,0.005,te) * smoothstep(0.05,0.,te);
    float v = wnoise*nenv + body;
    v = 0.5*v;
    v /= 1. + abs(v);
    
    //float click = noise(20000.*t) * exp(-600.*t);
    float click = sweep(te, 0.005);
    //v += click * 2.;
    
    v *= smoothstep(-1e-6,atk,te);
    v *= (1. + 0.3* smoothstep(0.01,0.0,te) + 0.2*smoothstep(0.05,0.2,te));
    
    return v;
}


vec2 tomDrum(float te, float f0, float df)
{
    float spd = 5.;
    // Instantaneous frequency
    float ifreq = f0 + df*exp(-te*spd);
    // Integrate to obtain
    float phase = f0*te + df/spd * (1.-exp(-te*spd));
    
    float env = smoothstep(0.,0.0005,te) * curve(te, 0.35, 0.0, -1.3);
    float noiseEnv = smoothstep(0.,0.0005,te) * curve(te, 0.8, 0.0, -2.) * (1. + 5.*curve(te, 0.022,0.0,0.));
    vec2 sig = vec2(0);
    
    sig += triBipolar(4.*phase) * env * 0.15;
    vec2 noise = (coloredNoise2(te, 1000., 20000.) + coloredNoise2(te, 4500., 6000.));
    
    sig += (noise.xx + noise.yy)*0.7 * noiseEnv * 0.03;
    sig /= 1. + abs(sig);
    sig += noise.yx * smoothstep(0.,0.05,te) * curve(te, 2.0, 0.0, -2.5) * 0.02; // Fake echo
    
    return sig;
}

float hihat(float t, float te)
{
    float sig = coloredNoise(te, 7500., 4500.) + coloredNoise(te, 2000., 1800.) * 0.1;
    //float env = smoothstep(0.0,0.0002,te) * (smoothstep(0.05,0.0,te) + 0.5*smoothstep(0.01,0.,te));
    float env = smoothstep(0.0,0.0002,te) * (curve(te, 0.2,0.0,-2.) + 0.5*smoothstep(0.01,0.,te));
    return sig * env;
}

vec2 clap(float t)
{
    // A "clap" sound effect
    vec2 sig = vec2(0);
    sig += (0.8*coloredNoise2(t, 1000., 800.) + 0.5*coloredNoise2(t, 3300.,3100.) + 0.4*coloredNoise2(t,8240.,8000.));
    
    sig *= smoothstep(0.,0.01,t) * (curve(t,1.0,0.03, -2.3) + 0.2*curve(t,5.0,0.1,-1.));
    // Fast envelope "stutter" is at the root of the "clap" sound
    sig *= mix(0.7, 0.5+0.5*sin(TAU*80.*t), smoothstep(0.06,0.03,t));
    sig /= 1.+abs(sig); // Distort
    
    return sig;
}

vec2 sawLead(float t, float te, float f)
{
    vec2 sig = vec2(0);
    te = max(te,0.);
    float env = exp(-5.*te) * smoothstep(0.,0.01,te);
    float fc0 = 2.*f;
    float fc = fc0 + (10000.-fc0)*exp(-8.*te);
    sig += lpfSaw3(t, f, fc, 1.);
    sig += vec2(-1,1) * lpfSaw3(t-0.05, f+1.618, fc, 2.);
    return sig * env * 0.1;
}

vec2 analogBrass(float t, float te, vec4 f)
{
    vec2 sig = vec2(0);
    float fc = 800. + 8000. * curve(te, 0.,0.3,-1.) * curve(te,2.,0.3,-2.); // Quick swell
    float env = smoothstep(0.,0.15,te); // Fade in only
    
    float amp = 1.;
    for(float i=0.; i<3.; i++){
        float dt = 0.06 * 0.05 * noise(t+5.62*i);
        sig += vec2(1.0,0.0) * lpfSaw3(t+dt, f.z, fc, 1.5) * amp;
        sig += vec2(0.8,0.6) * lpfSaw3(t+dt, f.x, fc, 1.5) * amp;
        sig += vec2(0.6,0.8) * lpfSaw3(t+dt, f.y, fc, 1.5) * amp;
        sig += vec2(0.0,1.0) * lpfSaw3(t+dt, f.w, fc, 1.5) * amp;
        amp *= 0.7;
    }
    
    vec2 warm = vec2(0);
    warm += vec2(1.0,0.0) * sin(TAU*f.z*t + sin(TAU*f.z*t));
    warm += vec2(0.8,0.6) * sin(TAU*f.x*t + sin(TAU*f.x*t));
    warm += vec2(0.6,0.8) * sin(TAU*f.y*t + sin(TAU*f.y*t));
    warm += vec2(0.0,1.0) * sin(TAU*f.w*t + sin(TAU*f.w*t));
    
    return sig * 0.05 + warm * 0.03;
}

vec2 fmLead(float te, float f1, float f2, float t1, float t2, float dur)
{
    // Lead synth sound
    
    // f1, f2, t1, t2 : portamento from f1 to f2 between time t1 and t2
    
    float phase = phasePortamento(te, f1, f2, t1, t2);
    vec2 phase2 = phase + te*vec2(-1,1);
    
    // vibrato
    float vibHz = 5.5;
    float vibAmp = smoothstep(0.5,1.2,te);
    float vibrato = sin(TAU*vibHz*te) * vibAmp;
    phase += 0.06*0.2*f1 * vibrato/(TAU*vibHz);
    
    float env = smoothstep(0.,0.01,te) * (1.3 - 0.5*smoothstep(0.,0.1,te) + 0.1*vibrato) * smoothstep(dur,dur-0.1,te);
    
    float iom = 5500./max(f1,f2);
    vec2 sig = vec2(0);
    sig += 3.*sin(TAU*phase + sin(TAU*phase2));
    sig += sin(TAU*phase*7. + (3.+vibrato)*sin(2.*TAU*phase2));
    sig += sin(TAU*(phase2+2.*te) + iom*sin(TAU*phase2));
    sig += sin(TAU*5000. + (5.+ vibrato)*sin(TAU*phase2.yx));
    
    return sig * env * 0.02;
}


vec2 shepardRiser(float t, float dt, float bandwidth)
{
    vec2 sig = vec2(0);
    
    for(float nn=30.; nn<100.; nn+=8.)
    {
        float midinote = nn + 15.*fract(t/dt);
        float fc = midicps(midinote);
        float df = bandwidth*fc;
        sig += coloredNoise2(mod(t,dt), fc, df) * 0.01;
    }
    
    return sig;
}

vec2 noiseRiseFall(float t, float dt)
{
    //float phase = smoothstep(0., dt, t) * dt;
    float x = t/dt;
    float phase = dt * x*x*x*(3.-2.25*x);
    vec2 sig = coloredNoise2(phase, 7000., 10000.) * 0.1;
    sig *= smoothstep(0., 0.01, t) * smoothstep(dt, dt-0.01, t);
    return sig;
}

vec2 machineButtonClick(float t)
{
    float reson = sin(TAU*805.*t + sin(TAU*302.*t) + sin(TAU*419.*t)) + 0.5*sin(TAU*100.*t);
    float noise = coloredNoise(t, 4393., 6000.) + 0.2*coloredNoise(t, 7000., 10000.);
    
    float sig = 0.;
    float env = percEnv(t, 0.001, 0.03, -1.) * 0.01 + percEnv(t-0.021, 0.001, 0.03, -1.) * 0.015
    + percEnv(t-0.045, 0.001, 0.04, -1.)*0.1 + percEnv(t-0.056, 0.001, 0.1, -1.)*0.15 + percEnv(t-0.09, 0.001, 0.3,-1.8) * 0.1;
    
    sig += (reson*0.1+noise) * env;
    
    return vec2(sig);
}

vec2 machineButtonClickVerb(float t)
{
    return machineButtonClick(t) + vec2(0.5,0)*machineButtonClick(t-0.0062) + vec2(0,0.3)*machineButtonClick(t-0.01);
}

vec2 crashCymbal(float t, float atk)
{
    float reson = sin(TAU*429.*t + 5.*sin(TAU*1120.*t) + 5.*sin(TAU*1812.*t));
    vec2 sig = coloredNoise2(t, 7150., 10000.) + 0.1*reson*smoothstep(0.,0.05,t);
    float env = curve(t, 15.0, 0.0,-3.) * curve(t, 0.0, 0.08, 1.);
    env *= (1. + smoothstep(0.02,0.0,t) * 2.);
    env *= (1. - smoothstep(0.0,0.05,t)*smoothstep(0.5,0.0,t) * 0.5);
    env *= smoothstep(0., atk, t);
    return sig * env * 0.2;
}

vec2 crashCymbalVerb(float t)
{
    vec2 sig = crashCymbal(t, 0.);
    sig += crashCymbal(t-0.75*beatdur, 0.05).yx * 0.5;
    sig += crashCymbal(t-1.50*beatdur, 0.10).xy * 0.25;
    return sig;
}



///////////////////////////////////
////////// PATTERNS ///////////////
///////////////////////////////////

vec2 drums(float t, float atk)
{
    // atk: zero, or greater for smoother signal attack.
    vec2 v = vec2(0);
    
    float te = mod(t, beatdur * 2.);
    v += kick(t, te, atk) * 0.1;
    
    te = mod(t-beatdur, 2.*beatdur);
    v += snare(t, te, atk) * 0.25;
    
    te = mod(t, beatdur * 0.5);
    float vel = 1. + mod(t,beatdur);
    vec2 panHH = 1. + 0.5 * vec2(-1,1) * cos(TAU*t/(4.*beatdur));
    v += hihat(t, te) * 0.03 * vel * panHH;
    
    return v * smoothstep(-1e-6,atk,t);
}

vec2 clapPattern(float t)
{
    vec2 v = vec2(0);
    float te = mod(t-7.*beatdur, 8.*beatdur);
    v += clap(te) * 0.25;
    
    return v * smoothstep(-1e-6,0.,t);
}

vec2 tomFill(float t)
{
    if(t < 0.5*beatdur || t > 3.*beatdur) return vec2(0);
    
    t = t - 0.5*beatdur;
    float te = mod(t, 0.75*beatdur);
    float num = floor(t / (0.75*beatdur));
    float f0i = 0.;
    float dfi = (200. - 20.*num)*2.;
    
    vec2 sig = vec2(0);
    sig += tomDrum(te, f0i, dfi);
    
    return sig;
}

vec2 pentatonicArp(float t)
{
    if(t < 0.) return vec2(0);
    float notedur = beatdur / 8.;
    float te = mod(t, notedur);
    float nthNote = floor(t / notedur);
    vec2 sig = vec2(0);
    for(float ni = 0.; ni < 8.; ni++) // Also play the tail of the previous notes
    {
        float nn = nthNote-ni;
        // fold back after 16 notes
        nn = fold(nn, 0., 16.);
        float degree = mod(nn, 5.); // degree in pentatonic scale
        float octave = floor(nn/5.);
        float note = floor(12.*degree/5.+0.6); // midi note number in pentatonic scale
        float midiNoteNum = 69. + 12.*octave + note;
        float f = midicps(midiNoteNum);

        float tei = te+notedur*ni;
        vec2 sigi = sawLead(t, tei, f);
        sigi *= smoothstep(notedur*8., notedur*7., tei); // fade out note to avoid clicks
        sig += sigi;
    }
    return sig;
}

vec2 pentatonicArpVerb(float t)
{
    vec2 pan = (1. + 0.7*vec2(-1,1)*sin(0.7*TAU*t));
    return pan * pentatonicArp(t) + vec2(0.7,0.3)*pentatonicArp(t-3./8.*beatdur-0.02)
     - vec2(0.2,0.5)*pentatonicArp(t-4./8.*beatdur-0.05);
}

vec2 bassLine(float t)
{
    float notedur = 0.5*beatdur;
    float te = mod(t, notedur);
    float nthNote = floor(t/notedur);
    nthNote = mod(nthNote + 1., 32.); // Start all chords one half note in advance
    
    float midiNoteNum = 
        (nthNote < 8.) ? 45. : // A
        (nthNote < 16.) ? 43. : // G major
        (nthNote < 28.) ? 48. : // C major
        50.; // D
    float freq = midicps(midiNoteNum);
    
    float pumping = smoothstep(0.,beatdur*0.5, mod(t,beatdur));
    vec2 sig = vec2(0);
    sig += bassStab(t, te, freq) * mix(1.,pumping,0.5) * 0.7;
    sig += bassStab(t, te, freq/2.) * mix(1.,pumping,0.6) * 0.7;
    
    return sig;
}

vec2 brassPad(float t)
{
    float nthNote = floor(t/beatdur);
    nthNote = mod(nthNote, 16.);
    vec4 midiNoteNum =
        (nthNote < 4.) ? vec4(60,62,64,69) : // A minor (+D)
        (nthNote < 8.) ? vec4(59,62,67,69) : // G
        (nthNote < 14.) ? vec4(60,62,64,67) : // C (+D)
        vec4(60,62,65,69); // Dm7
        
    vec2 env_startend = 
        (nthNote < 4.) ? vec2(0,4) : 
        (nthNote < 8.) ? vec2(4,8) :
        (nthNote < 14.) ? vec2(8,14) :
        vec2(14,16);
        
    float te = mod(t, 16.*beatdur) - env_startend.x * beatdur;
    float noteDur = (env_startend.y - env_startend.x) * beatdur;
    float fadeout = smoothstep(noteDur, noteDur-0.2, te);
    
    vec4 freq = midicps(midiNoteNum);
    
    vec2 sig = analogBrass(t, te, freq);
    sig *= fadeout;
    
    return sig;
}

vec2 brassPadVerb(float t)
{
    return brassPad(t) + 0.5*brassPad(t-0.1).yx + 0.2*brassPad(t-1.);
}

vec2 fmLeadPhrase(float t)
{
    if(t<0.) return vec2(0);
    
    t = mod(t, 32.*beatdur);
    if(t < 28.*beatdur) t = mod(t, 16.*beatdur); // variation on second time
    
    float tb = t / beatdur;
    
    // Each note may contain one portamento
    // Note data is (tb0, dur, midiNN1, midiNN2, beats until porta, porta duration in beats).
    mat3 noteData = 
        (tb < 2.5) ? mat3(0.0, 2.5, 74, 76, 0.5, 0.1, 0.0, 0.0, 0.0) : // D-E
        (tb < 3.5) ? mat3(2.5, 1, 74, 72, 0.5, 0.2, 0.0, 0.0, 0.0) :// D-C
        (tb < 6.5) ? mat3(3.5, 3, 79, 79, 10, 10, 0.0, 0.0, 0.0) : // G
        (tb < 8.5) ? mat3(7.5, 1, 81, 83, 0.0, 0.05, 0.0, 0.0, 0.0) : // B
        (tb < 9.5) ? mat3(8.5, 1, 84, 83, 0.5, 0.2, 0.0, 0.0, 0.0) : // C-B
        (tb < 11.) ? mat3(9.5, 1.5, 76,76,10,10, 0.0, 0.0, 0.0): // E
        (tb < 11.5) ? mat3(11., 0.5, 77,79,0.0,0.1, 0.0, 0.0, 0.0): // G
        (tb < 12.5) ? mat3(11.5, 1., 77,77,10,10, 0.0, 0.0, 0.0): // F
        (tb < 13.5) ? mat3(12.5, 1., 76,76,10,10, 0.0, 0.0, 0.0): // E
        (tb < 15.) ? mat3(13.5, 1.5, 76,74,0.0,0.05, 0.0, 0.0, 0.0): // E-D
        (tb < 16.) ? mat3(15., 1., 72,72,10,10, 0.0, 0.0, 0.0) : // C
        // identical repeat until...
        (tb < 29.5) ? mat3(27.5,2., 77,77,10,10, 0.0, 0.0, 0.0) : // F (continued)
        mat3(29.5, 1.7, 76,76,10,10, 0.0, 0.0, 0.0) // E
        ;
    
    
    float t0 = noteData[0][0]*beatdur;
    float te = t - t0;
    float dur = noteData[0][1]*beatdur;
    float f1 = midicps(noteData[1][0]);
    float f2 = midicps(noteData[1][1]);
    float t1 = noteData[2][0] * beatdur;
    float t2 = t1 + noteData[2][1] * beatdur;
    vec2 sig = fmLead(te, f1, f2, t1, t2, dur);
    
    return sig;
}

vec2 fmLeadPhraseVerb(float t)
{
    return fmLeadPhrase(t) + vec2(0.2,0.7) * fmLeadPhrase(t-beatdur+0.01) + vec2(0.5,0.1) * fmLeadPhrase(t-2.*beatdur-0.02);
}


/////////////////////////////////////
//////////// SONG STRUCTURE /////////
/////////////////////////////////////

vec2 introSequence(float t)
{
    // arps intro
    // +drums
    // +brass
    // +riser, tom fill
    
    float tb = t / beatdur;
    
    
    vec2 sig = vec2(0);
    
    float te =  mod(t,beatdur);
    float pumping = smoothstep(0.01,0.,te) + smoothstep(0.01,beatdur*0.5, te);
    float arpPump = 0.7*smoothstep(7.9,8.0,tb);
    sig += pentatonicArpVerb(t) * 0.09 * mix(1.,pumping,arpPump);
    
    t -= 8.*beatdur;
    vec2 drumSig = drums(t, 0.);
    sig /= 1. + 3.*abs(drumSig); // Distort the signal according to the drums
    sig += drumSig;
    
    
    pumping = smoothstep(0.01,0.,te) + smoothstep(0.01,beatdur*0.3, te);
    t -= 8.*beatdur;
    sig += brassPadVerb(t) * 0.4 * mix(1.,pumping,0.9)  * curve(t, 0., 24., 0.5);
    
    t -= 24.*beatdur;
    
    sig += noiseRiseFall(t, 7.*beatdur) * 0.2;
    // Play the first clap louder
    pumping = smoothstep(0.05,0.,te) + smoothstep(0.05,0.3, te);
    sig += clapPattern(t) * smoothstep(3.*beatdur,4.*beatdur,t) * (1. + 0.3*smoothstep(15.*beatdur, 8.*beatdur, t))
     * mix(1., pumping, 0.3);
    
    t -= 4.*beatdur;
    vec2 fill = tomFill(t);
    sig /= 1.+2.*abs(fill); // Distort signal according to toms
    sig += fill;
    
    return sig;
}


vec2 chorusPattern(float t)
{
    // Should be overlayed with intro sequence (t > 48.*beatdur)
    
    vec2 sig = vec2(0);
    
    float te =  mod(t,beatdur);
    float pumping = smoothstep(0.01,0.,te) + smoothstep(0.01,beatdur*0.5, te);
    
    sig += crashCymbalVerb(mod(t,32.*beatdur)) * 0.5 * mix(1., pumping, 0.5);
    sig += fmLeadPhraseVerb(t) * mix(1.,pumping, 0.8) * 0.7;
    sig += bassLine(t);
    
    return sig;
}

vec2 bootUp(float t)
{
    // Sequence of glitchy sounds
    // like a tape loading up.
    // Runs in 4 seconds
    
    vec2 sig = vec2(0);
    
    sig += machineButtonClickVerb(t-0.5);
    
    t -= 1.;
    
    float phase = rampIntegral(t / 3.) * 3.;
    
    vec2 sig1 = vec2(0);
    sig1 += 0.1 * pentatonicArpVerb(phase);
    sig1 += 0.02 * noise(5000.*phase);
    sig1 *= smoothstep(0.,0.1,t) * smoothstep(3.0,2.999,t);
    
    sig += machineButtonClick(1.05*(t-2.9));
    
    return sig + sig1 * 0.5;
}


vec2 fullSong(float t)
{
    vec2 sig = vec2(0);
    
    if(0. < t)
        sig += introSequence(t);
    if (12.*4.*beatdur < t)
    {
        t -= 12.*4.*beatdur;
        sig += chorusPattern(t);
    }
    
    return sig;
}

vec2 mainSound( float t )
{
    vec2 sig = vec2(0);
    
    float te =  mod(t,beatdur);
    float pumping = smoothstep(0.01,0.,te) + smoothstep(0.01,beatdur*0.5, te);

    
    
    sig += bootUp(t);
    
    t -= 4.;
    
    sig += fullSong(t);
    
    //sig /= 1. + 0.5 * abs(sig);
    
    
    return sig;
}`))

    //Send generated sound to speaker
    .pipe(Speaker());
    }
}
