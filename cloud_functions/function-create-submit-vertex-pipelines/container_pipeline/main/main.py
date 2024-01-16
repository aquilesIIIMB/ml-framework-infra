import json
import argparse


    # Add arguments
    parser.add_argument(
        "--project_id", help="BigQuery project", type=str, required=True
    )

    parser.add_argument(
        "--version",
        help="Execution version, could be a date or a name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--application_name", help="Application name", type=str, required=True
    )

    parser.add_argument("--model_name", help="Model name", type=str, required=True)

    parser.add_argument(
        "--pipeline_name", help="Pipeline name", type=str, required=True
    )

    parser.add_argument(
        "--endpoint_name",
        help="Vertex endpoint name to be deployed",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--service_account", help="Service Account", type=str, required=True
    )

    parser.add_argument(
        "--container_image_preprocessing",
        help="Container Image Preprocessing",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--container_image_training",
        help="Container Image Preprocessing",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--container_image_postprocessing",
        help="Container Image Preprocessing",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--container_image_deployment",
        help="Container Image Preprocessing",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--secret_path",
        help="Secret path in google secret manager",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--topic_monitoring_name",
        help="Monitoring Pub/Sub topic name",
        type=str,
        required=True,
    )

    parser.add_argument("--release", help="Release", type=str, default="")

    parser.add_argument("--git_url", help="git_url", type=str, default="")

    parser.add_argument("--git_branch", help="git_branch", type=str, default="")

    parser.add_argument("--git_commit", help="git_commit", type=str, default="")

    parser.add_argument(
        "--test_set_pct", help="Test set percent", type=float, required=True
    )

    parser.add_argument(
        "--validation_set_pct", help="Validation set percent", type=float, required=True
    )

    parser.add_argument(
        "--input_dataset_bq",
        help="BigQuery dataset to save model input data (preprocessing output)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dataset_bq",
        help="BigQuery dataset to save model output data (postprocessing output)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--metadata_dataset_bq",
        help="BigQuery dataset to save pipeline metadata",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--metadata_table_bq",
        help="BigQuery table to save pipeline metadata",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hp_training_iters_per_trial",
        help="Training iterations per trial in hyperparameters tuning process",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--hp_ntrials",
        help="Trial numbers to find distinct hyperparameters",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--hp_min_range_values",
        help="""Min value of range to search for hyperparameters
        (must have the same len to hp_names, hp_init_values and hp_max_range_values)""",
        action="append",
        type=float,
    )

    parser.add_argument(
        "--hp_max_range_values",
        help="""Max value of range to search for hyperparameters
        (must have the same len to hp_names, hp_init_values and hp_min_range_values)""",
        action="append",
        type=float,
    )

    parser.add_argument(
        "--hp_names",
        help="""Names of hyperparameters (must have the same len to hp_init_values,
        hp_min_range_values and hp_max_range_values)""",
        action="append",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hp_init_values",
        help="""Initial values of hyperparameters (must have the same len to hp_names,
        hp_min_range_values and hp_max_range_values)""",
        action="append",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--training_iters", help="Training iterations", type=int, required=True
    )

    parser.add_argument("--use_gpu", help="to use gpu or not", action="store_true")

    parser.add_argument(
        "--use_cache", help="to use cached components or not", action="store_true"
    )

    parser.add_argument(
        "--input_bucket_name",
        help="GCS bucket to save model input data (preprocessing output)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_bucket_name",
        help="GCS bucket to save model weights and files",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_bucket_name",
        help="GCS bucket to save model output data (postprocessing output)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--pipeline_bucket_name",
        help="GCS bucket to save pipeline artifacts and metadata",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prediction_bucket_name",
        help="GCS bucket to save model predictions and deployment metadata",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--preprocessing_data_quailty_decision_rule_name",
        help="Decision rule to assess preprocessing data quality (main metric name)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--preprocessing_data_quailty_decision_rule_value",
        help="Decision rule to assess preprocessing data quality (main metric value)",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--preprocessing_data_quailty_decision_rule_direction",
        help="""Decision rule to assess preprocessing data quality
        (main metric direction)[maximize or minimize]""",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--training_model_quailty_decision_rule_name",
        help="Decision rule to assess model quality (main metric name)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--training_model_quailty_decision_rule_value",
        help="Decision rule to assess model quality (main metric value)",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--training_model_quailty_decision_rule_direction",
        help="""Decision rule to assess model quality
        (main metric direction)[maximize or minimize]""",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--postprocessing_data_quailty_decision_rule_name",
        help="Decision rule to assess postprocessing data quality (main metric name)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--postprocessing_data_quailty_decision_rule_value",
        help="Decision rule to assess postprocessing data quality (main metric value)",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--postprocessing_data_quailty_decision_rule_direction",
        help="""Decision rule to assess postprocessing data quality
        (main metric direction)[maximize or minimize]""",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--drop_hp_tuning",
        help="to drop hyperparameter tuning process from pipeline execution",
        action="store_true",
    )

    parser.add_argument(
        "--drop_postprocessing",
        help="to drop postprocessing from pipeline execution",
        action="store_true",
    )

    parser.add_argument(
        "--drop_deployment",
        help="to drop deployment process from pipeline execution",
        action="store_true",
    )

    parser.add_argument(
        "--test_mode", help="Test mode to test pipeline execution", action="store_true"
    )

    parser.add_argument(
        "--schedule_cron",
        help="schedule cron to automate pipeline execution in a time interval",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--training_machine_cores",
        help="Machine cores for training process. Check compute engine machine types",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--training_machine_ram",
        help="Machine RAM fro training process. Check compute engine machine types",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--training_gpu_type",
        help="GPU type for training process. Check compute engine machine types",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--training_gpu_cores",
        help="GPU cores for training process. Check compute engine machine types",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--hp_machine_cores",
        help="""Machine cores for hyperparameters tuning process.
        Check compute engine machine types""",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hp_machine_ram",
        help="Machine RAM fro hyperparameters tuning process. Check compute engine machine types",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hp_gpu_type",
        help="GPU type for hyperparameters tuning process. Check compute engine machine types",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hp_gpu_cores",
        help="GPU cores for hyperparameters tuning process. Check compute engine machine types",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--endpoint_machine_type",
        help="Machine type for being deployed in Vertex Endpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--endpoint_min_replica_count",
        help="""The minimum number of replicas this DeployedModel
        will be always deployed on. If traffic against it increases,
        it may dynamically be deployed onto more replicas up to
        automatic_resources_max_replica_count, and as traffic decreases,
        some of these extra replicas may be freed. If the requested value
        is too large, the deployment will error.""",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--endpoint_max_replica_count",
        help="""The maximum number of replicas this DeployedModel may
        be deployed on when the traffic against it increases. If the requested
        value is too large, the deployment will error, but if deployment
        succeeds then the ability to scale the model to that many replicas
        is guaranteed (barring service outages). If traffic against the
        DeployedModel increases beyond what its replicas at maximum may handle,
        a portion of the traffic will be dropped.""",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pipeline_network",
        help="""The full name of the Google Compute Engine network to which the
        Pipeline should be peered.
        [Format]projects/{project}/global/networks/{network}.""",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--endpoint_network",
        help="""The full name of the Google Compute Engine network to which the
        Endpoint should be peered.
        [Format]projects/{project}/global/networks/{network}.""",
        type=str,
        default=None,
    )

    # ----------------------- INPUT -------------------------------------------------
    args = parser.parse_args()

def create_submit_vertex_pipelines(
    file_event: Dict, 
    context: Dict
) -> None:

    # Assess consistency and length of hyperparameters values
    hp_init_values_length = len(list(map(float, args.hp_init_values)))
    hp_min_range_values_length = len(list(map(float, args.hp_min_range_values)))
    hp_max_range_values_length = len(list(map(float, args.hp_max_range_values)))
    hp_names_length = len(args.hp_names)

    if hp_names_length != hp_init_values_length:
        raise ValueError(
            "hp_names and hp_init_values parameters do not have the same length"
        )

    if (not args.drop_hp_tuning) and (
        (hp_min_range_values_length != hp_max_range_values_length)
        or (hp_min_range_values_length != hp_names_length)
    ):
        raise ValueError(
            """hp_min_range_values, hp_max_range_values, hp_names and
                        hp_init_values parameters do not have the same length"""
        )

    # Assess allowed values for data quality metric direction
    if (args.preprocessing_data_quailty_decision_rule_direction != "maximize") and (
        args.preprocessing_data_quailty_decision_rule_direction != "minimize"
    ):
        raise ValueError(
            """preprocessing_data_quailty_decision_rule_direction
                        value must be maximize or minimize"""
        )
    if (args.training_model_quailty_decision_rule_direction != "maximize") and (
        args.training_model_quailty_decision_rule_direction != "minimize"
    ):
        raise ValueError(
            """training_model_quailty_decision_rule_direction
                        value must be maximize or minimize"""
        )
    if (args.postprocessing_data_quailty_decision_rule_direction != "maximize") and (
        args.postprocessing_data_quailty_decision_rule_direction != "minimize"
    ):
        raise ValueError(
            """postprocessing_data_quailty_decision_rule_direction
                         value must be maximize or minimize"""
        )

    # Save parameters in a file
    PIPELINE_PARAMETERS = vars(args)
    with open("./config.py", "w") as file:
        file.write("PIPELINE_PARAMETERS = ")
        file.write(
            json.dumps(PIPELINE_PARAMETERS, indent=4)
            .replace("false", "False")
            .replace("true", "True")
        )
        file.write("\n")

    from pipeline_builder import build_pipeline

    # Pipeline execution
    build_pipeline()
