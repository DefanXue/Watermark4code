private void firePropertyChanged(List listeners, PropertyEvent event) {
   if (listeners == null || listeners.isEmpty()) {
      return;
   }

   for (PropertyListener listener : listeners) {
      if (listener != null) {
         listener.propertyChanged(event);
      }
   }
}