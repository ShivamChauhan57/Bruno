package com.example.brunoface

import android.Manifest
import android.os.Bundle
import android.view.WindowManager
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.example.brunoface.databinding.ActivityMainBinding
import org.json.JSONObject

class MainActivity : ComponentActivity() {
  private lateinit var binding: ActivityMainBinding
  private lateinit var tts: TtsManager
  private lateinit var ws: WsClient
  private lateinit var rtc: WebRtcClient

  private val reqPerms = registerForActivityResult(
    ActivityResultContracts.RequestMultiplePermissions()
  ) { _ -> startEverything() }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)
    goImmersive()

    val wv: WebView = binding.webView
    wv.settings.javaScriptEnabled = true
    wv.settings.domStorageEnabled = true
    wv.settings.cacheMode = WebSettings.LOAD_NO_CACHE
    wv.webChromeClient = WebChromeClient()
    wv.loadUrl("file:///android_asset/face.html")

    reqPerms.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
  }

  private fun startEverything(){
    tts = TtsManager(this); tts.init()

    // TODO: change to your PC’s LAN IP (e.g., ws://192.168.1.23:8765)
    val pcWsUrl = "ws://PC_LAN_IP:8765"
    ws = WsClient(pcWsUrl) { handleIncoming(it) }
    ws.connect()

    rtc = WebRtcClient(this) { msg -> ws.send(msg) }
    rtc.start()
  }

  private fun handleIncoming(txt: String){
    try{
      val obj = JSONObject(txt)
      when(obj.optString("type")){
        "answer" -> rtc.onAnswer(obj.optString("sdp"))
        "ice"    -> rtc.onRemoteIce(obj.getJSONObject("candidate"))
        else -> {
          obj.optJSONObject("tts")?.optString("text")?.let { if (it.isNotEmpty()) tts.speak(it) }
          runOnUiThread {
            val jsonStr = obj.toString()
            binding.webView.evaluateJavascript("updateFace(${jsonStr.quoteJs()})", null)
          }
        }
      }
    }catch(_: Exception){}
  }

  private fun goImmersive() {
    WindowCompat.setDecorFitsSystemWindows(window, false)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    val controller = WindowInsetsControllerCompat(window, findViewById(android.R.id.content))
    controller.hide(WindowInsetsCompat.Type.systemBars())
    controller.systemBarsBehavior =
      WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
  }

  override fun onResume() { super.onResume(); goImmersive() }
  override fun onDestroy() { super.onDestroy(); tts.shutdown() }
}

private fun String.quoteJs(): String =
  this.replace("\\", "\\\\").replace("\"", "\\\"")
