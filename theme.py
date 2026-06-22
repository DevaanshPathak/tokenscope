TOKEN_COLORS = [
    "#49d6e8",
    "#f2cc60",
    "#d779ff",
    "#74a7ff",
    "#ffb15c",
    "#7ee787",
    "#ff8cc6",
    "#f0f6fc",
]

APP_CSS = """
Screen {
    background: #0d1117;
    color: #c9d1d9;
}

.hidden {
    display: none;
}

#header {
    dock: top;
    height: 1;
    padding: 0 1;
    background: #161b22;
    color: #f0f6fc;
}

#path-screen {
    margin: 1 2;
    padding: 1 2;
    border: round #30363d;
    height: auto;
}

#path-title {
    text-style: bold;
    color: #f0f6fc;
    margin-bottom: 1;
}

#path-input {
    margin-top: 1;
}

#path-status {
    height: auto;
    min-height: 1;
    margin-top: 1;
    color: #8b949e;
}

#loading {
    margin-top: 1;
}

#main-layout {
    height: 1fr;
    layout: vertical;
}

#text-input {
    height: 3;
    margin: 1 1 0 1;
    border: round #30363d;
}

#content-row {
    height: 3fr;
    min-height: 10;
}

TokenView {
    width: 2fr;
    margin: 1 0 0 1;
    padding: 1;
    border: round #30363d;
    overflow-y: auto;
}

StatsPanel {
    width: 1fr;
    min-width: 32;
    margin: 1 1 0 1;
    padding: 1;
    border: round #30363d;
}

MergeTreeWidget {
    height: 2fr;
    min-height: 12;
    margin: 1;
    padding: 0 1 1 1;
    border: round #30363d;
}

TabbedContent {
    height: 1fr;
}

TabPane {
    padding: 1 0 0 0;
}

#token-table,
#vocab-table {
    height: 1fr;
}

#merge-tree {
    height: 1fr;
    overflow-y: auto;
}

#vocab-search {
    height: 3;
    margin-bottom: 1;
}
"""
