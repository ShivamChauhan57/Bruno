package com.example.brunoface

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.speech.tts.TextToSpeech
import java.util.Locale

class TtsManager(private val ctx: Context): TextToSpeech.OnInitListener {
  private var tts: TextToSpeech? = null
  private var ready = false

  fun init(){ tts = TextToSpeech(ctx, this) }
  override fun onInit(status: Int) {
    if (status==TextToSpeech.SUCCESS){
      tts?.language = Locale.US
      tts?.setSpeechRate(1.0f)
      ready = true
    }
  }

  fun speak(text: String){
    if(!ready) return
    val am = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    am.mode = AudioManager.MODE_IN_COMMUNICATION
    am.isSpeakerphoneOn = true
    tts?.setAudioAttributes(
      AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()
    )
    tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, System.currentTimeMillis().toString())
  }

  fun shutdown(){ tts?.shutdown() }
}
