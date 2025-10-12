protected void HandleOnRefreshing(boolean var_45) {
    ProcessSmoothScrollTo(PerformGetFooterSize(), null);
    ProcessSmoothScrollTo(-PerformGetHeaderSize(), null);
    PerformCallRefreshListener();
    ProcessSmoothScrollTo(0, null);

    if (val_54 != null) {
        val_54.refreshing();
    }
    if (var_38 != null) {
        var_38.refreshing();
    }

    if (var_73 != null) {
        var_73.showHeaderLoadingLayout();
        var_73.showFooterLoadingLayout();
    }

    if (var_45) {
        // Intentionally left blank to preserve functional parity.
    }
}