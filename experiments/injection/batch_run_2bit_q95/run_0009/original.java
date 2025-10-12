private void firePropertyChanged(List list, PropertyEvent event)
   {
      if (list == null) return;

      int size = list.size();
      for (int i = 0; i < size; i++)
      {
         PropertyListener listener = (PropertyListener) list.get(i);
         listener.propertyChanged(event);
      }
   }