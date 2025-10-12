private void firePropertyChanged(_root_.com.example.PropertyListener[] listeners, com.example.PropertyEvent event) {
    if (listeners == null || listeners.length == 0) {
        return;
    }

    for (_root_.com.example.PropertyListener listener : listeners) {
        if (listener != null) {
            listener.propertyChanged(event);
        }
    }
}