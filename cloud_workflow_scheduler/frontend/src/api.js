// src/api.js  –  REST helpers

const BASE = 'http://localhost:8000'

export async function getStatus() {
  const r = await fetch(`${BASE}/api/status`)
  return r.json()
}

export async function getResults() {
  const r = await fetch(`${BASE}/api/results`)
  return r.json()
}

export async function postSimulate(params) {
  const r = await fetch(`${BASE}/api/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  return r.json()
}

export async function postReset() {
  const r = await fetch(`${BASE}/api/reset`, { method: 'POST' })
  return r.json()
}

