package com.example.brunoface

import okhttp3.*
import okio.ByteString
import org.json.JSONObject

class WsClient(private val url: String,
               private val onText: (String)->Unit): WebSocketListener(){
  private var ws: WebSocket? = null

  fun connect(){
    val client = OkHttpClient()
    val req = Request.Builder().url(url).build()
    ws = client.newWebSocket(req, this)
  }

  fun send(obj: JSONObject){ ws?.send(obj.toString()) }

  override fun onOpen(webSocket: WebSocket, response: Response) {
    webSocket.send(JSONObject().put("role","face").toString())
  }
  override fun onMessage(webSocket: WebSocket, text: String) { onText(text) }
  override fun onMessage(webSocket: WebSocket, bytes: ByteString) {}
}
