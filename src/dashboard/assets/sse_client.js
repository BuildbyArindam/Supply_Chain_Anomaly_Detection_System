/* Live alerts over Server-Sent Events (Day 18).
 *
 * Opens an EventSource on the URL in #sse-config[data-sse-url] and writes every alert into the
 * dcc.Store "sse-latest" ({alert, received_at, count}). The browser reconnects by itself and
 * resumes from the last event id; on "resync" or "overflow" the Interval-driven stores already
 * hold the full picture, so nothing more is needed here.
 */
(function () {
  "use strict";
  var source = null;
  var count = 0;

  function setStatus(text, ok) {
    var el = document.getElementById("sse-status");
    if (el) {
      el.textContent = "Live alerts: " + text;
      el.style.color = ok ? "#00bc8c" : "#9a9a9a";
    }
  }

  function publish(alert) {
    count += 1;
    var props = { data: { alert: alert, received_at: Date.now(), count: count } };
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props("sse-latest", props);
    }
  }

  function connect(url) {
    if (!window.EventSource) {
      setStatus("not supported", false);
      return;
    }
    source = new EventSource(url);
    source.addEventListener("hello", function () { setStatus("connected", true); });
    source.addEventListener("alert", function (e) {
      try { publish(JSON.parse(e.data)); } catch (err) { console.warn("bad SSE alert", err); }
    });
    source.addEventListener("resync", function () { setStatus("resyncing", true); });
    source.onerror = function () { setStatus("reconnecting…", false); };
  }

  var tries = 0;
  var timer = setInterval(function () {
    var cfg = document.getElementById("sse-config");
    tries += 1;
    if (cfg) {
      clearInterval(timer);
      var url = cfg.getAttribute("data-sse-url");
      if (url) { connect(url); } else { setStatus("disabled", false); }
    } else if (tries > 200) {
      clearInterval(timer);
    }
  }, 100);
})();
