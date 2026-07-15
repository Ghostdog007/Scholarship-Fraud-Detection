// Same-origin by default: the browser only ever talks to nginx, which serves
// this frontend AND reverse-proxies /v3, /health, /ready, /docs to the API.
// Because everything is one origin, no CORS is involved and no host is
// hard-coded — this works identically on localhost, on the server's IP, and
// behind a k8s Ingress hostname. Leave this empty unless you deliberately
// split the frontend onto a different origin from the API.
const API_BASE = "";
