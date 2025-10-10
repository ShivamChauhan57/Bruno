package com.example.brunoface

import android.content.Context
import org.json.JSONObject
import org.webrtc.*

class WebRtcClient(
  private val ctx: Context,
  private val wsSend: (JSONObject)->Unit
){
  private lateinit var pc: PeerConnection
  private lateinit var factory: PeerConnectionFactory
  private var videoSource: VideoSource? = null
  private var audioSource: AudioSource? = null
  private var capturer: CameraVideoCapturer? = null

  fun start(){
    PeerConnectionFactory.initialize(
      PeerConnectionFactory.InitializationOptions.builder(ctx).createInitializationOptions()
    )
    val opts = PeerConnectionFactory.Options()
    val egl = EglBase.create()
    val encoder = DefaultVideoEncoderFactory(egl.eglBaseContext, true, true)
    val decoder = DefaultVideoDecoderFactory(egl.eglBaseContext)
    factory = PeerConnectionFactory.builder()
      .setOptions(opts).setVideoEncoderFactory(encoder).setVideoDecoderFactory(decoder).build()

    val iceServers = listOf(PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer())
    pc = factory.createPeerConnection(PeerConnection.RTCConfiguration(iceServers),
      object: PeerConnection.Observer{
        override fun onIceCandidate(c: IceCandidate){
          val msg = JSONObject().put("type","ice").put("candidate", JSONObject()
            .put("sdpMid", c.sdpMid).put("sdpMLineIndex", c.sdpMLineIndex).put("candidate", c.sdp))
          wsSend(msg)
        }
        override fun onConnectionChange(newState: PeerConnection.PeerConnectionState) {}
        override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {}
        override fun onSignalingChange(p0: PeerConnection.SignalingState) {}
        override fun onStandardizedIceConnectionChange(p0: PeerConnection.IceConnectionState) {}
        override fun onAddStream(p0: MediaStream?) {}
        override fun onDataChannel(p0: DataChannel?) {}
        override fun onIceCandidatesRemoved(p0: Array<out IceCandidate>?) {}
        override fun onRemoveStream(p0: MediaStream?) {}
        override fun onRenegotiationNeeded() {}
        override fun onAddTrack(p0: RtpReceiver?, p1: Array<out MediaStream>?) {}
        override fun onTrack(transceiver: RtpTransceiver?) {}
      })!!

    val surfaceHelper = SurfaceTextureHelper.create("CaptureThread", EglBase.create().eglBaseContext)
    videoSource = factory.createVideoSource(false)
    val enumerator = Camera2Enumerator(ctx)
    val front = enumerator.deviceNames.firstOrNull { enumerator.isFrontFacing(it) }
    capturer = enumerator.createCapturer(front ?: enumerator.deviceNames.first(), null)
    capturer?.initialize(surfaceHelper, ctx, videoSource!!.capturerObserver)
    capturer?.startCapture(640, 480, 24)
    pc.addTrack(factory.createVideoTrack("v0", videoSource))

    audioSource = factory.createAudioSource(MediaConstraints())
    pc.addTrack(factory.createAudioTrack("a0", audioSource))

    val mc = MediaConstraints().apply {
      mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio","false"))
      mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo","false"))
    }
    pc.createOffer(object: SdpObserver {
      override fun onCreateSuccess(desc: SessionDescription) {
        pc.setLocalDescription(object: SdpObserver{ override fun onSetSuccess(){}; override fun onSetFailure(p0:String){}; override fun onCreateSuccess(p0: SessionDescription){}; override fun onCreateFailure(p0:String){} }, desc)
        wsSend(JSONObject().put("type","offer").put("sdp", desc.description))
      }
      override fun onSetSuccess(){}
      override fun onCreateFailure(p0: String){}
      override fun onSetFailure(p0: String){}
    }, mc)
  }

  fun onAnswer(sdp: String){
    pc.setRemoteDescription(object: SdpObserver{ override fun onSetSuccess(){}; override fun onSetFailure(p0:String){}; override fun onCreateSuccess(p0: SessionDescription){}; override fun onCreateFailure(p0:String){} },
      SessionDescription(SessionDescription.Type.ANSWER, sdp))
  }
  fun onRemoteIce(candidate: JSONObject){
    val c = IceCandidate(candidate.optString("sdpMid"), candidate.optInt("sdpMLineIndex"), candidate.optString("candidate"))
    pc.addIceCandidate(c)
  }
}
