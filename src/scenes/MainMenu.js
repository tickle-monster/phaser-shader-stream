let AudioShader = require('audio-shader');
let Speaker = require('audio-speaker/stream');
let glslify = require('glslify');
let jukebox, marshall;

import { Scene } from 'phaser';

export class MainMenu extends Scene
{
    constructor ()
    {
        super('MainMenu');
    }

    create ()
    {
        this.add.image(512, 384, 'background');

        this.add.image(512, 200, 'logo');

        this.add.text(512, 260, 'Click or tap anywhere to start streaming\na GLSL audio file adapted from Shadertoy.com', {
            fontFamily: 'Arial Black', fontSize: 24, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'center'
        }).setOrigin(0.5,0);


        //marshall = new Speaker();

        [1,2,3,4,5].forEach( tracknumber => {
            this.add.text(62+tracknumber*150, 360, 'GLSL\nFile '+tracknumber, {
                fontFamily: 'Arial Black', fontSize: 24, color: '#22ff22',
                stroke: '#000000', strokeThickness: 8, align: 'center'
                }).setOrigin(0.5,0).setInteractive().on('pointerdown', () => {
                    console.log('jukebox',jukebox);
                    if (jukebox) {
                        jukebox = AudioShader(glslify("")).pipe(Speaker());
//                        jukebox = new AudioShader(glslify(this.cache.text.get('music'+tracknumber))).pipe(Speaker());
                    } //else {
                        jukebox = AudioShader(glslify(this.cache.text.get('music'+tracknumber))).pipe(Speaker());
                    console.log('jukebox2',jukebox);
                    //}
                    //console.log('AudioShader2',AudioShader());
                    //AudioShader(glslify(this.cache.text.get('music'+tracknumber))).pipe(Speaker());
                    //console.log('AudioShader3',AudioShader());
                });
        });

        this.add.text(512, 460, 'Stop playing', {
            fontFamily: 'Arial Black', fontSize: 24, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'center'
        }).setOrigin(0.5,0).setInteractive().on('pointerdown', () => {
                console.log('AudioShader4',AudioShader());
                console.log('Speaker4',Speaker());
                //AudioShader.destroy;
                console.log('jukebox3',jukebox);
                if (jukebox) {
                    //var newbuffer = new Phaser.Sound.WebAudioSound();
                    //jukebox.drain();
                    jukebox.end(jukebox); }
                //jukebox.drain();
                console.log('AudioShader5',AudioShader());
                console.log('jukebox4',jukebox);
            });
    }
}
