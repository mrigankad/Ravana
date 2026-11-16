"""Ravana icon set: 24px grid, 1.75 stroke, round caps/joins, 2px live-area padding.

Style follows Lucide/Feather conventions (pairs well with Inter). Colour comes from
``currentColor`` so the app can tint icons; ink #15120F is the default.

Every icon is listed with the surface that needs it (see design/kit/KIT.md).
"""

from __future__ import annotations

INK = "#15120F"


def svg(body: str, color: str = INK, stroke: float = 1.75) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


DOT = 'fill="{c}" stroke="none"'.format(c=INK)

ICONS: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------- navigation (title bar tabs)
    "swap": {"group": "Navigation", "body": '<circle cx="7" cy="9" r="3.5"/><circle cx="17" cy="15" r="3.5"/><path d="M13 5h4a2 2 0 0 1 2 2v2"/><path d="M17 7l2 2 2-2"/><path d="M11 19H7a2 2 0 0 1-2-2v-2"/><path d="M7 17l-2-2-2 2"/>'},
    "batch": {"group": "Navigation", "body": '<rect x="7" y="7" width="13" height="13" rx="2"/><path d="M4 16V6a2 2 0 0 1 2-2h10"/><circle cx="13.5" cy="12" r="2"/><path d="M9.5 20c.6-2 2.2-3 4-3s3.4 1 4 3"/>'},
    "live": {"group": "Navigation", "body": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10.5l5-3v9l-5-3"/><circle cx="7" cy="10" r="1.2" ' + DOT + '/>'},
    "models": {"group": "Navigation", "body": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5"/><path d="M12 12v9"/>'},
    "history": {"group": "Navigation", "body": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v4h4"/><path d="M12 8v4l3 2"/>'},
    "doctor": {"group": "Navigation", "body": '<path d="M3 12h4l2-5 4 10 2-5h6"/>'},
    "settings": {"group": "Navigation", "body": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1"/>'},

    # ---------------------------------------------------------- inputs (drop cards, file dialogs)
    "source-face": {"group": "Inputs", "body": '<circle cx="12" cy="9" r="4"/><path d="M5 20c1.2-3.5 3.8-5 7-5s5.8 1.5 7 5"/>'},
    "target-image": {"group": "Inputs", "body": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 9"/>'},
    "target-video": {"group": "Inputs", "body": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9.5v5l4.5-2.5z"/>'},
    "drop": {"group": "Inputs", "body": '<path d="M12 4v11"/><path d="M7.5 10.5L12 15l4.5-4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>'},
    "folder": {"group": "Inputs", "body": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'},
    "folder-open": {"group": "Inputs", "body": '<path d="M3 17V7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v1"/><path d="M3 17l2.5-6h16L19 17z"/>'},
    "replace": {"group": "Inputs", "body": '<path d="M4 9a8 8 0 0 1 14-3l2 2"/><path d="M20 4v4h-4"/><path d="M20 15a8 8 0 0 1-14 3l-2-2"/><path d="M4 20v-4h4"/>'},

    # ---------------------------------------------------------- settings (rail rows)
    "quality": {"group": "Settings", "body": '<path d="M12 3l2.5 5.5L20 9l-4 4 1 6-5-3-5 3 1-6-4-4 5.5-.5z"/>'},
    "restore": {"group": "Settings", "body": '<circle cx="11" cy="13" r="7"/><path d="M8.5 12h.01M13.5 12h.01"/><path d="M8.5 15.5c1.4 1.2 3.6 1.2 5 0"/><path d="M19 2v4M17 4h4"/>'},
    "pixel-boost": {"group": "Settings", "body": '<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/><path d="M13 7h4a2 2 0 0 1 2 2v1"/><path d="M11 17H7a2 2 0 0 1-2-2v-1"/>'},
    "faces": {"group": "Settings", "body": '<circle cx="12" cy="12" r="4"/><path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3"/>'},
    "faces-multi": {"group": "Settings", "body": '<circle cx="8" cy="10" r="3"/><circle cx="16.5" cy="10" r="3"/><path d="M3 19c.8-2.4 2.7-3.5 5-3.5s4.2 1.1 5 3.5"/><path d="M13.5 16c.8-.4 1.8-.5 3-.5 2.3 0 4.2 1.1 5 3.5"/>'},
    "gpu": {"group": "Settings", "body": '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>'},
    "cpu": {"group": "Settings", "body": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>'},
    "mask": {"group": "Settings", "body": '<path d="M12 3c4.5 0 7 3 7 7.5S16 21 12 21 5 15 5 10.5 7.5 3 12 3z"/><path d="M5.5 13h13"/><path d="M9 9h.01M15 9h.01"/>'},
    "lighting": {"group": "Settings", "body": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'},
    "temporal": {"group": "Settings", "body": '<path d="M3 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/><path d="M3 18h18" opacity=".5"/>'},
    "sliders": {"group": "Settings", "body": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>'},

    # ---------------------------------------------------------- stage (canvas toolbar)
    "compare": {"group": "Stage", "body": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 2v20"/><path d="M8 10l-2 2 2 2"/><path d="M16 10l2 2-2 2"/>'},
    "side-by-side": {"group": "Stage", "body": '<rect x="3" y="5" width="8" height="14" rx="1.5"/><rect x="13" y="5" width="8" height="14" rx="1.5"/>'},
    "result": {"group": "Stage", "body": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 12l3 3 5-6"/>'},
    "zoom-in": {"group": "Stage", "body": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M11 8v6M8 11h6"/>'},
    "zoom-out": {"group": "Stage", "body": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M8 11h6"/>'},
    "fit": {"group": "Stage", "body": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>'},
    "save": {"group": "Stage", "body": '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/>'},
    "reveal": {"group": "Stage", "body": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M10 14h5M13 12l2 2-2 2"/>'},
    "copy": {"group": "Stage", "body": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>'},
    "variants": {"group": "Stage", "body": '<circle cx="6" cy="12" r="3"/><circle cx="12" cy="12" r="3"/><circle cx="18" cy="12" r="3"/>'},

    # ---------------------------------------------------------- status & metrics (readout, banners)
    "score": {"group": "Status & metrics", "body": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/><circle cx="12" cy="17" r="1.3"/>'},
    "identity": {"group": "Status & metrics", "body": '<path d="M7 4.5A9 9 0 0 1 20.5 12"/><path d="M4 9a8.5 8.5 0 0 0 0 6"/><path d="M8 18a6 6 0 0 1-1-6 5 5 0 0 1 9.5-.5"/><path d="M12 12c0 3-1 5.5-2.5 7.5"/><path d="M16.5 14c-.2 2.3-.9 4.4-2 6"/>'},
    "sharpness": {"group": "Status & metrics", "body": '<circle cx="12" cy="12" r="3"/><path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3"/>'},
    "timer": {"group": "Status & metrics", "body": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M10 2h4"/>'},
    "check": {"group": "Status & metrics", "body": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>'},
    "warning": {"group": "Status & metrics", "body": '<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17h.01"/>'},
    "error": {"group": "Status & metrics", "body": '<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>'},
    "info": {"group": "Status & metrics", "body": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'},
    "spinner": {"group": "Status & metrics", "body": '<path d="M12 3a9 9 0 1 0 9 9"/>'},
    "no-face": {"group": "Status & metrics", "body": '<circle cx="12" cy="12" r="7"/><path d="M9.5 10.5h.01M14.5 10.5h.01"/><path d="M9.5 15h5"/><path d="M3 3l18 18"/>'},

    # ---------------------------------------------------------- batch (queue rows, toolbar)
    "add-file": {"group": "Batch", "body": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M12 12v6M9 15h6"/>'},
    "add-folder": {"group": "Batch", "body": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 10v6M9 13h6"/>'},
    "trash": {"group": "Batch", "body": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'},
    "retry": {"group": "Batch", "body": '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>'},
    "play": {"group": "Batch", "body": '<path d="M7 4.5v15l12-7.5z"/>'},
    "pause": {"group": "Batch", "body": '<path d="M8 5v14M16 5v14"/>'},
    "stop": {"group": "Batch", "body": '<rect x="6" y="6" width="12" height="12" rx="2"/>'},
    "queue": {"group": "Batch", "body": '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1" ' + DOT + '/><circle cx="4" cy="12" r="1" ' + DOT + '/><circle cx="4" cy="18" r="1" ' + DOT + '/>'},
    "sort": {"group": "Batch", "body": '<path d="M7 4v16M3 16l4 4 4-4"/><path d="M14 6h7M14 12h5M14 18h3"/>'},

    # ---------------------------------------------------------- live (camera tab)
    "camera": {"group": "Live", "body": '<path d="M4 8a2 2 0 0 1 2-2h2l1.5-2h5L16 6h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="3.5"/>'},
    "camera-off": {"group": "Live", "body": '<path d="M9.5 4h5L16 6h2a2 2 0 0 1 2 2v8.5M17 20H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2"/><path d="M3 3l18 18"/>'},
    "record": {"group": "Live", "body": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4" fill="#E0401F" stroke="none"/>'},
    "virtual-cam": {"group": "Live", "body": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10.5l5-3v9l-5-3"/><path d="M7 12h5M10 10l2 2-2 2"/>'},
    "fps": {"group": "Live", "body": '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>'},

    # ---------------------------------------------------------- models (download, verify, licence)
    "download": {"group": "Models", "body": '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>'},
    "verified": {"group": "Models", "body": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9 12l2 2 4-4"/>'},
    "licence": {"group": "Models", "body": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>'},
    "hash": {"group": "Models", "body": '<path d="M5 9h14M5 15h14M10 4L8 20M16 4l-2 16"/>'},
    "storage": {"group": "Models", "body": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'},

    # ---------------------------------------------------------- trust (consent, provenance, privacy)
    "consent": {"group": "Trust", "body": '<path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V11"/><path d="M11 10.5V4.5a1.5 1.5 0 0 1 3 0v6"/><path d="M14 10.5v-4a1.5 1.5 0 0 1 3 0V14c0 4-2.5 7-6 7-2.5 0-4-1-5.5-3L4 15.5a1.5 1.5 0 0 1 2.5-1.8L8 15"/>'},
    "watermark": {"group": "Trust", "body": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9 11.5c1-1 2-1 3 0s2 1 3 0"/>'},
    "provenance": {"group": "Trust", "body": '<circle cx="12" cy="9" r="6"/><path d="M9 14.5L8 21l4-2 4 2-1-6.5"/><path d="M9.5 9l1.8 1.8L15 7.5"/>'},
    "local-only": {"group": "Trust", "body": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>'},
    "eye-off": {"group": "Trust", "body": '<path d="M3 3l18 18"/><path d="M10.6 5.1A9.8 9.8 0 0 1 12 5c5 0 8.5 4.5 9.5 7a13 13 0 0 1-2.6 3.8M6.6 6.6A13 13 0 0 0 2.5 12c1 2.5 4.5 7 9.5 7a9.6 9.6 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>'},

    # ---------------------------------------------------------- utility
    "chevron-down": {"group": "Utility", "body": '<path d="M6 9l6 6 6-6"/>'},
    "chevron-right": {"group": "Utility", "body": '<path d="M9 6l6 6-6 6"/>'},
    "close": {"group": "Utility", "body": '<path d="M18 6L6 18M6 6l12 12"/>'},
    "more": {"group": "Utility", "body": '<circle cx="5" cy="12" r="1" ' + DOT + '/><circle cx="12" cy="12" r="1" ' + DOT + '/><circle cx="19" cy="12" r="1" ' + DOT + '/>'},
    "keyboard": {"group": "Utility", "body": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>'},
    "external": {"group": "Utility", "body": '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>'},
    "terminal": {"group": "Utility", "body": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M12 15h5"/>'},
    "docs": {"group": "Utility", "body": '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5M8 7h7M8 11h5"/>'},
    "code": {"group": "Utility", "body": '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>'},
}

# Used by the README feature table and the app's SelectRow chevron.
ALIASES = {"chevron": "chevron-down"}
