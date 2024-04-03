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

        this.add.image(612, 384, 'background');

        this.add.image(612, 300, 'logo');

        this.add.text(612, 460, 'You are now listening to\na GLSL audio file\nadapted from Shadertoy.com', {
            fontFamily: 'Arial Black', fontSize: 38, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8,
            align: 'center'
        }).setOrigin(0.5);

    AudioShader(glslify(this.cache.text.get('music1')))
        .pipe(Speaker());
    }
}
