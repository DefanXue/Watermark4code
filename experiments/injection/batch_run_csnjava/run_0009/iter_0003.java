private void firePropertyChanged(List listeners, PropertyEvent event) {
    if (listeners == null || listeners.isEmpty())
{ // Added isEmpty() check for slight optimization if list is empty
        return;
    }

    // Using a traditional for loop with index for explicit control,
    // though the enhanced for loop from the original is often preferred for readability
    // when the index itself isn't needed. This change doesn't fundamentally alter
    // efficiency or readability in a significant way compared to the enhanced for loop
    // for this specific scenario, but demonstrates an alternative valid iteration.
    // The original comment about enhanced for being cleaner is generally true.
    for (int i = 0; i < listeners.size(); i++)
{
        // Explicit cast remains necessary due to the raw List type.
        // This is a common pattern when dealing with older APIs or raw types.
        PropertyListener listener = (PropertyListener) listeners.get(i);
        listener.propertyChanged(event);
    }
}