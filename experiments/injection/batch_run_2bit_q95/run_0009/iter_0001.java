private void firePropertyChanged(java.util.List<_root_.com.example.PropertyListener> list, com.example.PropertyEvent event) {
    if (list == null) {
        return;
    }

    int size = list.size();
    for (int i = 0; i < size; i++) {
        _root_.com.example.PropertyListener listener = (_root_.com.example.PropertyListener) list.get(i);
        listener.propertyChanged(event);
    }
}