from typing import NamedTuple, List

from kfp.v2.dsl import (
    Input,
    Metrics,
    Output,
    component,
    Dataset,
    Artifact,
    OutputPath,
    Markdown,
    PipelineTaskFinalStatus,
)

# from utils.deploy import deploy_pipeline
from config import PIPELINE_PARAMETERS


# Create Preprocessing component -------------------------------------------------------------
@component(
    base_image=PIPELINE_PARAMETERS.get("container_image_preprocessing"),
    output_component_file="preprocessing.yaml",
)
def preprocessing(
    version: str,
    model_name: str,
    secret_path: str,
    test_set_pct: float,
    validation_set_pct: float,
    project_id: str,
    input_dataset_bq: str,
    input_bucket_name: str,
    test_mode: bool,
    pipeline_labels: dict,
) -> NamedTuple("InputDataPath", [("input_table_bq_uri_list", List[str])]):
    """
    Ingest data from bigquery tables and generate features to train model.

    Parameters
    ----------
    version: str
        Featrues, model and output model version.
    model_name: str
        Model name to identify and save model artifact.
    secret_path: str
        Secret path in Google Cloud Secret Manager where are model secrets.
    test_set_pct: float
        Data percent to test model and get performance metrics.
    validation_set_pct: float
        Data percent to validate model and fit hyperparameters or choose model architecture.
    project_id: str
        Google Cloud project id.
    input_dataset_bq: str
        Bigquery dataset name where are data source tables.
    input_bucket_name: str
        Cloud Storage bucket name where are data source files.
    test_mode: bool
        If we want to test pipeline execution or not.
    pipeline_labels: dict
        Labels to identify artifacts, jobs, pipeline, Bigquery tables and Cloud Storage files.

    Returns
    ------
    Tuple with features tables list.
    """
    import logging
    from collections import namedtuple
    from model_preprocessing import PipeLine

    # Running container pipeline
    preprocessing_pipeline = PipeLine(
        version=version,
        model_name=model_name,
        test_set_pct=test_set_pct,
        validation_set_pct=validation_set_pct,
        project_id=project_id,
        input_dataset_bq=input_dataset_bq,
        input_bucket_name=input_bucket_name,
        secret_path=secret_path,
        pipeline_labels=pipeline_labels,
        test_mode=test_mode,
    )
    output_preprocessing = preprocessing_pipeline.run_pipeline()
    input_table_bq_uri_list = []
    for output in output_preprocessing:
        input_table_bq_uri_list.append(f"bq://{output}")

    logging.info("\n".join(input_table_bq_uri_list))

    input_data_path = namedtuple("InputDataPath", ["input_table_bq_uri_list"])

    logging.info("Generated preprocessing tables.")

    return input_data_path(input_table_bq_uri_list)


# Create Preprocessing Validation component --------------------------------------------------
@component(
    base_image=PIPELINE_PARAMETERS.get("container_image_preprocessing"),
    output_component_file="preprocess_validation.yaml",
)
def preprocessing_stats_validation(
    preprocessing_data_quailty_decision_rule_name: str,
    preprocessing_data_quailty_decision_rule_value: float,
    preprocessing_data_quailty_decision_rule_direction: str,
    input_table_bq_uri: str,
    preprocessing_stats_json: OutputPath("Dataset"),
    preprocessing_stats_reshaped_markdown: Output[Markdown],
):
    """
    Assess data quality for model features.

    Parameters
    ----------
    preprocessing_data_quailty_decision_rule_name: str
        Statistical metric name to assess data quality in features table.
        ex: density, selectivity, outliers_count, nulls_count, max_value, min_value
    preprocessing_data_quailty_decision_rule_value: float
        Statistical metric thershold to assess data quality in features table.
    preprocessing_data_quailty_decision_rule_direction: str
        If the best value is achieved by maximizing or minimizing the statistical metric.
        Allowed values: maximize or minimize.
    input_table_bq_uri: str
        Bigquery table URI where are data source.

    Returns
    ------
    preprocessing_stats_json: OutputPath("Dataset")
        Dataset artifact to store statistical information of features using json format
        in Cloud Storage.
    preprocessing_stats_reshaped_markdown: Output[Markdown]
        Markdown artifact to store summary statistical information of features using
        markdown format.
    """
    import json
    import logging
    import pandas as pd
    from model_preprocessing_validation import PipeLine
    from utils import downsampling_stats

    # Parameters
    nbins = 1000
    ncolumns_per_iteration = 10

    logging.info("input_table_bq_uri: " + input_table_bq_uri)

    project_id, input_dataset_bq, input_table_bq = input_table_bq_uri.replace(
        "bq://", ""
    ).split(".")

    # Running container pipeline
    preprocessing_validation_pipeline = PipeLine(
        project_id=project_id,
        input_dataset_bq=input_dataset_bq,
        input_table_bq=input_table_bq,
        nbins=nbins,
        ncolumns_per_iteration=ncolumns_per_iteration,
    )
    preprocessing_stats_dict = preprocessing_validation_pipeline.run_pipeline()

    # Save stats as a dict in a Dataset Artifact
    with open(preprocessing_stats_json, "w") as f:
        f.write(json.dumps(preprocessing_stats_dict))

    df_preprocessing_stats = pd.DataFrame.from_dict(preprocessing_stats_dict)

    # Downsampling from near 1000 bins to 10 bins
    df_preprocessing_stats_reshaped = downsampling_stats(df_preprocessing_stats)

    logging.info(df_preprocessing_stats_reshaped.to_markdown())

    # Save stats downsampled to Markdown Artifact
    with open(preprocessing_stats_reshaped_markdown.path, "w") as f:
        f.write(df_preprocessing_stats_reshaped.to_markdown())

    # Decision rule to assess data quality
    if preprocessing_data_quailty_decision_rule_direction == "maximize":
        decision_rule = (
            int(
                df_preprocessing_stats[
                    preprocessing_data_quailty_decision_rule_name
                ].min()
            )
            < preprocessing_data_quailty_decision_rule_value
        )
    else:
        decision_rule = (
            int(
                df_preprocessing_stats[
                    preprocessing_data_quailty_decision_rule_name
                ].max()
            )
            > preprocessing_data_quailty_decision_rule_value
        )

    if decision_rule:
        raise ValueError("Issue in Preprocessing Data Quality")

    logging.info("Estimated preprocessing stats.")


# Create Hyperparameter Tuning component -----------------------------------------------------
@component(
    base_image=PIPELINE_PARAMETERS.get("container_image_training"),
    output_component_file="hp_tuning.yaml",
)
def hp_tuning(
    version: str,
    model_name: str,
    hp_training_iters_per_trial: int,
    hp_ntrials: int,
    hp_min_range_values: List[float],
    hp_max_range_values: List[float],
    hp_names: List[str],
    hp_init_values: List[float],
    use_gpu: bool,
    model_bucket_name: str,
    drop_hp_tuning: bool,
    input_table_bq_uri_list: List[str],
    hyperparameters: Output[Metrics],
):
    """
    Choose a set of optimal hyperparameters for model architecture training.

    Parameters
    ----------
    version: str
        Featrues, model and output model version.
    model_name: str
        Model name to identify and save model artifact.
    hp_training_iters_per_trial: int
        Training iterations or epocs per trial. A trial is the process to assess
        performance of a hyperparametes set.
    hp_ntrials: int
        Number of hyperparameters sets to assess. A trial is the process to assess
        performance of a hyperparametes set.
    hp_min_range_values: List[float]
        A list of min values for each hyperparameter. Fist list element is the first
        model hyperparameter and last list element is the last model hyperparameter.
    hp_max_range_values: List[float]
        A list of max values for each hyperparameter. Fist list element is the first
        model hyperparameter and last list element is the last model hyperparameter.
    hp_names: List[str]
        A list of the model hyperparameters names. Fist list element is the first
        model hyperparameter and last list element is the last model hyperparameter.
    hp_init_values: List[float]
        A list of default values for each hyperparameter. Fist list element is the first
        model hyperparameter and last list element is the last model hyperparameter.
    use_gpu: bool
        If we want to train the model using GPU or not.
    model_bucket_name: str
        Cloud Storage bucket name where are saved model files.
    drop_hp_tuning: bool
        If we want to drop hyperparameter tuning process or not.
    input_table_bq_uri_list: List[str]
        Bigquery table URI list where are data sources.

    Returns
    ------
    hyperparameters: Output[Metrics]
        Metric artifact where are optimal hyperparameters or default hyperparameters
        (when hp tuning is dropped). Hyperparameters are saved in json format.
    """
    import json
    import logging

    if drop_hp_tuning:
        best_hp = dict(zip(hp_names, hp_init_values))

    else:
        from model_hp_tuning import PipeLine_hp

        # Running container pipeline
        hp_tuning_pipeline = PipeLine_hp(
            version=version,
            model_name=model_name,
            hp_training_iters_per_trial=hp_training_iters_per_trial,
            hp_ntrials=hp_ntrials,
            hp_min_range_values=hp_min_range_values,
            hp_max_range_values=hp_max_range_values,
            hp_names=hp_names,
            hp_init_values=hp_init_values,
            use_gpu=use_gpu,
            model_bucket_name=model_bucket_name,
            input_table_bq_uri_list=input_table_bq_uri_list,
        )
        performance_metric, best_hp = hp_tuning_pipeline.run_pipeline()

    logging.info(f"Performance metrics: {json.dumps(performance_metric)}")
    logging.info(f"Hyperparameters: {json.dumps(best_hp)}")

    # Save hyperparameters in a metric artifact
    for key, value in best_hp.items():
        hyperparameters.log_metric(key, value)

    logging.info("Found hyperparameters.")


# Create Training component ------------------------------------------------------------------
@component(
    base_image=PIPELINE_PARAMETERS.get("container_image_training"),
    output_component_file="training.yaml",
)
def training(
    version: str,
    model_name: str,
    training_iters: int,
    use_gpu: bool,
    model_bucket_name: str,
    training_model_quailty_decision_rule_name: str,
    training_model_quailty_decision_rule_value: float,
    training_model_quailty_decision_rule_direction: str,
    input_table_bq_uri_list: List[str],
    hyperparameters: Input[Metrics],
    model_uri: OutputPath(str),
    performance_metrics: Output[Metrics],
):
    """
    Weight optimization of an ML/AI architecture.

    Parameters
    ----------
    version: str
        Featrues, model and output model version.
    model_name: str
        Model name to identify and save model artifact.
    training_iters: int
        Training iterations or epocs to optimize weights
    use_gpu: bool
        If we want to train the model using GPU or not.
    model_bucket_name: str
        Cloud Storage bucket name where are model files.
    training_model_quailty_decision_rule_name: str
        Statistical metric name to assess model quality.
        ex: r2, rmse, rank_pct, auc, accuracy
    training_model_quailty_decision_rule_value: float
        Statistical metric thershold to assess model quality.
    training_model_quailty_decision_rule_direction: str
        If the best value is achieved by maximizing or minimizing the statistical metric.
        Allowed values: maximize or minimize.
    input_table_bq_uri_list: List[str]
        Bigquery table URI list where are data sources.
    hyperparameters: Input[Metrics]
        Metric artifact where are optimal hyperparameters or default hyperparameters
        (when hp tuning is dropped). Hyperparameters are saved in json format.

    Returns
    ------
    model_uri: OutputPath(str)
        Model Cloud Storage URI where is saved the model.
    performance_metrics: Output[Metrics]
        Metric artifact where are model performance metrics. Metrics are saved in json format.
    """
    import json
    import logging
    from model_training import PipeLine

    # Running container pipeline
    training_pipeline = PipeLine(
        version=version,
        model_name=model_name,
        hyperparameters=hyperparameters.metadata,
        training_iters=training_iters,
        use_gpu=use_gpu,
        is_saved=True,
        model_bucket_name=model_bucket_name,
        input_table_bq_uri_list=input_table_bq_uri_list,
    )
    model_folder_path, _, metrics = training_pipeline.run_pipeline()

    logging.info(f"Performance metrics: {json.dumps(metrics)}")

    # Save performance metrics in a metric artifact
    for key, value in metrics.items():
        performance_metrics.log_metric(key, value)

    # Save model file in a GCS path
    with open(model_uri, "w") as f:
        f.write(f"gs://{model_folder_path}")

    # Decision rule
    if training_model_quailty_decision_rule_direction == "maximize":
        decision_rule = (
            metrics[training_model_quailty_decision_rule_name]
            < training_model_quailty_decision_rule_value
        )
    else:
        decision_rule = (
            metrics[training_model_quailty_decision_rule_name]
            > training_model_quailty_decision_rule_value
        )

    if decision_rule:
        raise ValueError("Issue in Model Performance Metric")

    logging.info("Trained model.")


# Create Postprocessing component ------------------------------------------------------------
if not PIPELINE_PARAMETERS.get("DROP_POSTPROCESSING"):

    @component(
        base_image=PIPELINE_PARAMETERS.get("container_image_postprocessing"),
        output_component_file="postprocessing.yaml",
    )
    def postprocessing(
        version: str,
        model_name: str,
        project_id: str,
        output_dataset_bq: str,
        output_bucket_name: str,
        test_mode: bool,
        secret_path: str,
        pipeline_labels: dict,
        model_uri: str,
    ) -> NamedTuple("OutputDataPath", [("output_table_bq_uri_list", List[str])]):
        """
        Model output data from bigquery tables using metadata or model information as source.

        Parameters
        ----------
        version: str
            Featrues, model and output model version.
        model_name: str
            Model name to identify and save model artifact.
        project_id: str
            Google Cloud project id.
        output_dataset_bq: str
            Bigquery dataset name where are model output (metadata or model information
            not predictions) tables.
        output_bucket_name: str
            Cloud Storage bucket name where are model output (metadata or model information
            not predictions) files.
        test_mode: bool
            If we want to test pipeline execution or not.
        secret_path: str
            Secret path in Google Cloud Secret Manager where are model secrets.
        pipeline_labels: dict
            Labels to identify artifacts, jobs, pipeline, Bigquery tables and
            Cloud Storage files.
        model_uri: str
            Model Cloud Storage URI where is saved the model.

        Returns
        ------
        Tuple with model output (metadata or model information not predictions) tables list.
        """
        import logging
        from collections import namedtuple
        from model_postprocessing import PipeLine

        # Running container pipeline
        postprocessing_pipeline = PipeLine(
            version=version,
            model_name=model_name,
            project_id=project_id,
            output_dataset_bq=output_dataset_bq,
            output_bucket_name=output_bucket_name,
            test_mode=test_mode,
            secret_path=secret_path,
            pipeline_labels=pipeline_labels,
            model_uri=model_uri,
        )
        output_postprocessing = postprocessing_pipeline.run_pipeline()
        output_table_bq_uri_list = []
        for output in output_postprocessing:
            output_table_bq_uri_list.append(f"bq://{output}")

        logging.info("\n".join(output_table_bq_uri_list))

        output_data_path = namedtuple("OutputDataPath", ["output_table_bq_uri_list"])

        logging.info("Generated postprocessing tables.")

        return output_data_path(output_table_bq_uri_list)


# Create Postprocessing Validation component -------------------------------------------------
if not PIPELINE_PARAMETERS.get("DROP_POSTPROCESSING"):

    @component(
        base_image=PIPELINE_PARAMETERS.get("container_image_postprocessing"),
        output_component_file="postprocess_validation.yaml",
    )
    def postprocessing_stats_validation(
        postprocessing_data_quailty_decision_rule_name: str,
        postprocessing_data_quailty_decision_rule_value: float,
        postprocessing_data_quailty_decision_rule_direction: str,
        output_table_bq_uri: str,
        postprocessing_stats_json: OutputPath("Dataset"),
        postprocessing_stats_reshaped_markdown: Output[Markdown],
    ):
        """
        Assess data quality for model output data.

        Parameters
        ----------
        postprocessing_data_quailty_decision_rule_name: str
            Statistical metric name to assess data quality in model output table.
            ex: density, selectivity, outliers_count, nulls_count, max_value, min_value
        postprocessing_data_quailty_decision_rule_value: float
            Statistical metric thershold to assess data quality in model output table.
        postprocessing_data_quailty_decision_rule_direction: str
            If the best value is achieved by maximizing or minimizing the statistical metric.
            Allowed values: maximize or minimize.
        output_table_bq_uri: str
            Bigquery table URI where are model output data.

        Returns
        ------
        postprocessing_stats_json: OutputPath("Dataset")
            Dataset artifact to store statistical information of model output data using json
            format in Cloud Storage.
        postprocessing_stats_reshaped_markdown: Output[Markdown]
            Markdown artifact to store summary statistical information of model output data
            using markdown format.
        """
        import json
        import logging
        import pandas as pd
        from model_postprocessing_validation import PipeLine
        from utils import downsampling_stats

        nbins = 1000
        ncolumns_per_iteration = 10

        # Users Latent Vectors Stats
        project_id, dataset_bq, table_bq = output_table_bq_uri.replace(
            "bq://", ""
        ).split(".")

        postprocessing_validation_pipeline = PipeLine(
            project_id=project_id,
            output_dataset_bq=dataset_bq,
            output_table_bq=table_bq,
            nbins=nbins,
            ncolumns_per_iteration=ncolumns_per_iteration,
        )
        postprocessing_stats_values = postprocessing_validation_pipeline.run_pipeline()

        with open(postprocessing_stats_json, "w") as f:
            f.write(json.dumps(postprocessing_stats_values))

        df_postprocessing_stats = pd.DataFrame.from_dict(postprocessing_stats_values)

        # Downsampling from near 1000 bins to 10 bins
        df_postprocessing_stats_reshaped = downsampling_stats(df_postprocessing_stats)

        logging.info(df_postprocessing_stats_reshaped.to_markdown())

        # Save reshaped postprocessing stats to markdown artifact
        with open(postprocessing_stats_reshaped_markdown.path, "w") as f:
            f.write(df_postprocessing_stats_reshaped.to_markdown())

        # Decision rule
        if postprocessing_data_quailty_decision_rule_direction == "maximize":
            decision_rule = (
                int(
                    df_postprocessing_stats[
                        postprocessing_data_quailty_decision_rule_name
                    ].min()
                )
                < postprocessing_data_quailty_decision_rule_value
            )
        else:
            decision_rule = (
                int(
                    df_postprocessing_stats[
                        postprocessing_data_quailty_decision_rule_name
                    ].max()
                )
                > postprocessing_data_quailty_decision_rule_value
            )

        if decision_rule:
            raise ValueError("Issue in Postprocessing Data Quality")

        logging.info("Estimated postprocessing stats.")


# Create Deployment component ----------------------------------------------------------------
if not PIPELINE_PARAMETERS.get("DROP_DEPLOYMENT"):

    @component(
        base_image=PIPELINE_PARAMETERS.get("container_image_deployment"),
        output_component_file="deployment.yaml",
    )
    def pre_deployment(
        endpoint_name: str,
        project_id: str,
        region: str,
        pipeline_bucket_name: str,
        pipeline_labels: dict,
        network: str,
        vertex_endpoint: Output[Artifact],
        request_schema: OutputPath("Dataset"),
        response_schema: OutputPath("Dataset"),
    ):
        """
        Reuse of existing Endpoint ID or creating a new Endpoint. In the case of using a
        private Endpoint (when a shared VPC is used), the models deployed on the existing
        Endpoint are undeployed.

        Parameters
        ----------
        endpoint_name: str
            Endpoint name to identify endpoint of the deployed models. It's not unique.
        project_id: str
            Google Cloud project id.
        region: str
            Data and machines location to process jobs.
        pipeline_bucket_name: str
            Cloud Storage bucket to save pipeline artifacts and metadata.
        pipeline_labels: dict
            Labels to identify artifacts, jobs, pipeline, Bigquery tables and
            Cloud Storage files.
        network: str
            The full name of the Google Compute Engine network to which the Endpoint
            should be peered.

        Returns
        ------
        vertex_endpoint: Output[Artifact]
            Endpoint artifact with the endpoint URI and metadata.
        request_schema: OutputPath("Dataset")
            Dataset artifact with the Endpoint request schema to get a model prediction.
        response_schema: OutputPath("Dataset")
            Dataset artifact with the Endpoint response schema to save a model prediction.
        """
        import time
        import json
        import logging
        import google.cloud.aiplatform as aip

        from schemas import Request, Response

        pipeline_root = f"gs://{pipeline_bucket_name}/pipeline_root"
        aip.init(project=project_id, location=region, staging_bucket=pipeline_root)

        if network:
            # List Vertex Endpoint
            aip_endpoint_list = aip.PrivateEndpoint.list(
                filter=f"display_name=private-{endpoint_name}",
                order_by="create_time desc",
                project=project_id,
                location=region,
            )

            if aip_endpoint_list:
                aip_endpoint = aip_endpoint_list[0]
                if "deployedModels" in aip_endpoint.to_dict().keys():
                    for models in aip_endpoint.to_dict()["deployedModels"]:
                        aip_endpoint.undeploy(deployed_model_id=models["id"])

                time.sleep(120)

                vertex_endpoint.metadata["resourceName"] = aip_endpoint.resource_name
                vertex_endpoint.uri = f"https://{region}-aiplatform.googleapis.com/{aip_endpoint.resource_name}"  # noqa: E501

            else:
                aip_endpoint = aip.PrivateEndpoint.create(
                    display_name=f"private-{endpoint_name}",
                    project=project_id,
                    location=region,
                    network=network,
                    labels=pipeline_labels,
                )
                vertex_endpoint.metadata["resourceName"] = aip_endpoint.resource_name
                vertex_endpoint.uri = f"https://{region}-aiplatform.googleapis.com/{aip_endpoint.resource_name}"  # noqa: E501

        else:
            # List Vertex Endpoint
            aip_endpoint_list = aip.Endpoint.list(
                filter=f"display_name={endpoint_name}",
                order_by="create_time desc",
                project=project_id,
                location=region,
            )

            if aip_endpoint_list:
                aip_endpoint = aip_endpoint_list[0]
                vertex_endpoint.metadata["resourceName"] = aip_endpoint.resource_name
                vertex_endpoint.uri = f"https://{region}-aiplatform.googleapis.com/{aip_endpoint.resource_name}"  # noqa: E501

            else:
                aip_endpoint = aip.Endpoint.create(
                    project=project_id,
                    display_name=endpoint_name,
                    location=region,
                    labels=pipeline_labels,
                )
                vertex_endpoint.metadata["resourceName"] = aip_endpoint.resource_name
                vertex_endpoint.uri = f"https://{region}-aiplatform.googleapis.com/{aip_endpoint.resource_name}"  # noqa: E501

        with open(request_schema, "w") as f:
            f.write(json.dumps(Request.schema()))

        with open(response_schema, "w") as f:
            f.write(json.dumps(Response.schema()))

        logging.info("Deployed API endpoint.")


# Create Post deployment component -----------------------------------------------------------
if not PIPELINE_PARAMETERS.get("DROP_DEPLOYMENT"):

    @component(
        base_image=PIPELINE_PARAMETERS.get("container_image_deployment"),
        output_component_file="post_deployment.yaml",
    )
    def post_deployment(
        endpoint_name: str,
        project_id: str,
        region: str,
        network: str,
        pipeline_bucket_name: str,
    ):
        """
        Model output data from bigquery tables using metadata or model information as source.

        Parameters
        ----------
        endpoint_name: str
            Endpoint name to identify endpoint of the deployed models. It's not unique.
        project_id: str
            Google Cloud project id.
        region: str
            Data and machines location to process jobs.
        network: str
            The full name of the Google Compute Engine network to which the Endpoint
            should be peered.
        pipeline_bucket_name: str
            Cloud Storage bucket to save pipeline artifacts and metadata.
        """
        import logging
        import google.cloud.aiplatform as aip

        pipeline_root = f"gs://{pipeline_bucket_name}/pipeline_root"
        aip.init(project=project_id, location=region, staging_bucket=pipeline_root)

        if not network:
            # List Vertex Endpoint
            aip_endpoint_list = aip.Endpoint.list(
                filter=f"display_name={endpoint_name}",
                order_by="create_time desc",
                project=project_id,
                location=region,
            )

            aip_endpoint = aip_endpoint_list[0]
            for models in aip_endpoint.to_dict()["deployedModels"]:
                if models["id"] not in list(
                    aip_endpoint.to_dict()["trafficSplit"].keys()
                ):
                    aip_endpoint.undeploy(deployed_model_id=models["id"])

        logging.info("Cleaned API endpoint.")


# Create Monitoring Trigger component --------------------------------------------------------
@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-pubsub"],
    output_component_file="monitoring_trigger.yaml",
)
def monitoring_trigger(
    project_id: str,
    topic_monitoring_name: str,
    pipeline_id: str,
    pipeline_name: str,
    pipeline_resource: str,
    pipeline_type: str,
    pipeline_labels: dict,
    application_name: str,
    model_name: str,
    endpoint_name: str,
    region: str,
    secret_path: str,
    input_bucket_name: str,
    model_bucket_name: str,
    output_bucket_name: str,
    pipeline_bucket_name: str,
    prediction_bucket_name: str,
    test_set_pct: float,
    validation_set_pct: float,
    input_dataset_bq: str,
    output_dataset_bq: str,
    hp_training_iters_per_trial: int,
    hp_ntrials: int,
    training_iters: int,
    use_gpu: bool,
    schedule_cron: str,
    service_account: str,
    container_image_preprocessing: str,
    container_image_training: str,
    container_image_postprocessing: str,
    container_image_deployment: str,
    metadata_dataset_bq: str,
    metadata_table_bq: str,
    version: str,
    release: str,
    git_url: str,
    git_branch: str,
    git_commit: str,
    preprocessing_data_quailty_decision_rule_name: str,
    preprocessing_data_quailty_decision_rule_value: float,
    preprocessing_data_quailty_decision_rule_direction: str,
    training_model_quailty_decision_rule_name: str,
    training_model_quailty_decision_rule_value: float,
    training_model_quailty_decision_rule_direction: str,
    postprocessing_data_quailty_decision_rule_name: str,
    postprocessing_data_quailty_decision_rule_value: float,
    postprocessing_data_quailty_decision_rule_direction: str,
    drop_hp_tuning: bool,
    drop_postprocessing: bool,
    drop_deployment: bool,
    test_mode: bool,
    pipeline_network: str,
    endpoint_network: str,
    status: PipelineTaskFinalStatus,
):
    """
    Gather pipeline metadata and information and send it through a pub/sub topic message.

    """
    if not test_mode:
        import json
        import logging
        from google.cloud import pubsub

        publisher = pubsub.PublisherClient()

        topic_path = publisher.topic_path(project_id, topic_monitoring_name)

        record = {
            "project_id": project_id,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "pipeline_resource": pipeline_resource,
            "pipeline_status": status.state,
            "pipeline_type": pipeline_type,
            "pipeline_labels": pipeline_labels,
            "application_name": application_name,
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "region": region,
            "secret_path": secret_path,
            "input_bucket_name": input_bucket_name,
            "model_bucket_name": model_bucket_name,
            "output_bucket_name": output_bucket_name,
            "pipeline_bucket_name": pipeline_bucket_name,
            "prediction_bucket_name": prediction_bucket_name,
            "test_set_pct": test_set_pct,
            "validation_set_pct": validation_set_pct,
            "input_dataset_bq": input_dataset_bq,
            "output_dataset_bq": output_dataset_bq,
            "hp_training_iters_per_trial": hp_training_iters_per_trial,
            "hp_ntrials": hp_ntrials,
            "training_iters": training_iters,
            "use_gpu": use_gpu,
            "schedule_cron": schedule_cron,
            "service_account": service_account,
            "container_image_preprocessing": container_image_preprocessing,
            "container_image_training": container_image_training,
            "container_image_postprocessing": container_image_postprocessing,
            "container_image_deployment": container_image_deployment,
            "metadata_dataset_bq": metadata_dataset_bq,
            "metadata_table_bq": metadata_table_bq,
            "version": version,
            "release": release,
            "git_url": git_url,
            "git_branch": git_branch,
            "git_commit": git_commit,
            "preprocessing_data_quailty_decision_rule_name": (
                preprocessing_data_quailty_decision_rule_name
            ),
            "preprocessing_data_quailty_decision_rule_value": (
                preprocessing_data_quailty_decision_rule_value
            ),
            "preprocessing_data_quailty_decision_rule_direction": (
                preprocessing_data_quailty_decision_rule_direction
            ),
            "training_model_quailty_decision_rule_name": (
                training_model_quailty_decision_rule_name
            ),
            "training_model_quailty_decision_rule_value": (
                training_model_quailty_decision_rule_value
            ),
            "training_model_quailty_decision_rule_direction": (
                training_model_quailty_decision_rule_direction
            ),
            "postprocessing_data_quailty_decision_rule_name": (
                postprocessing_data_quailty_decision_rule_name
            ),
            "postprocessing_data_quailty_decision_rule_value": (
                postprocessing_data_quailty_decision_rule_value
            ),
            "postprocessing_data_quailty_decision_rule_direction": (
                postprocessing_data_quailty_decision_rule_direction
            ),
            "drop_hp_tuning": drop_hp_tuning,
            "drop_postprocessing": drop_postprocessing,
            "drop_deployment": drop_deployment,
            "test_mode": test_mode,
            "pipeline_network": pipeline_network,
            "endpoint_network": endpoint_network,
        }
        message = json.dumps(record)

        logging.info(message)

        # Data must be a bytestring
        data = message.encode("utf-8")

        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data=data)

        logging.info(future)

        logging.info("Published messages.")
