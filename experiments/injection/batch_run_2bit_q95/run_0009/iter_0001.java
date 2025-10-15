private void firePropertyChanged(List listeners, PropertyEvent event) {
   if (listeners == null) return;

   int size = listeners.size();
   for (int i = 0; i < size; i++) {
      PropertyListener listener = (PropertyListener) listeners.get(i);
      listener.propertyChanged(event);
   }
}