'use strict';
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('desktop', Object.freeze({
  request: (action, payload = {}) => ipcRenderer.invoke('request', action, payload),
  chooseFolder: () => ipcRenderer.invoke('choose-folder'),
  exportResults: rows => ipcRenderer.invoke('export-results', rows),
  importMetadata: () => ipcRenderer.invoke('import-metadata'),
  openPath: file => ipcRenderer.invoke('open-path', file),
  onEvent: callback => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on('worker-event', listener);
    return () => ipcRenderer.removeListener('worker-event', listener);
  }
}));
