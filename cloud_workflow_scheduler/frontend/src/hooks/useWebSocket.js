// src/hooks/useWebSocket.js
// Real-time WebSocket hook that reconnects automatically

import { useEffect, useRef, useCallback, useState } from 'react'

const WS_URL = `ws://${window.location.hostname}:8000/ws/live-stream`

export function useWebSocket(onMessage) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const reconnectTimeout = useRef(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessageRef.current(data)
      } catch (_) {}
    }

    ws.onclose = () => {
      setConnected(false)
      reconnectTimeout.current = setTimeout(connect, 2000)
    }

    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected }
}

