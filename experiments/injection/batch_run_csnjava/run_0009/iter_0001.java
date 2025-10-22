private void firePropertyChanged(List list, PropertyEvent event) {
    if (list == null) {
        return;
    }

    // Using an enhanced for loop for cleaner iteration
    // This implicitly handles the size and index, making the code more readable
    // and less error-prone regarding loop bounds.
    for (Object item : list) {
        // Explicit cast is still necessary as the List is raw (List instead of List<PropertyListener>)
        PropertyListener listener = (PropertyListener) item;
        listener.propertyChanged(event);
    }
}