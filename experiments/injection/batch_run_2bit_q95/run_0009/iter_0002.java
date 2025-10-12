private void firePropertyChanged(_root_.com.example.PropertyListener[] listeners, com.example.PropertyEvent event) {
    if (listeners == null) {
        return;
    }

    for (_root_.com.example.PropertyListener listener : listeners) {
        listener.propertyChanged(event);
    }
}