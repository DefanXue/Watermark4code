private void firePropertyChanged(List listeners, PropertyEvent event) {
   if (listeners == null || listeners.isEmpty()) {
      return;
   }

   for (java.util.Iterator it = listeners.iterator(); it.hasNext();) {
      PropertyListener listener = (PropertyListener) it.next();
      if (listener != null) {
         listener.propertyChanged(event);
      }
   }
}