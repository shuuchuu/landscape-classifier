def entraine_cli() -> None:
    import dvc.api

    from classifieur_paysages.entrainement import entraine

    entraine(**dvc.api.params_show(stages=["entrainement"]))
