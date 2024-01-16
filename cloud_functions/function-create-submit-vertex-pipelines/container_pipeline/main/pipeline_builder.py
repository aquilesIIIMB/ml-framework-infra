from typing import List

import kfp
from kfp.v2 import compiler, dsl
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components import aiplatform as gcc_aip

from utils.deploy import deploy_pipeline
from config import PIPELINE_PARAMETERS
from pipeline_components import (
    preprocessing,
    preprocessing_stats_validation,
    hp_tuning,
    training,
    monitoring_trigger,
)

if not PIPELINE_PARAMETERS.get("drop_postprocessing"):
    from pipeline_components import postprocessing, postprocessing_stats_validation
if not PIPELINE_PARAMETERS.get("drop_deployment"):
    from pipeline_components import pre_deployment, post_deployment


def build_pipeline():
    # ----------------------- PIPELINE -----------------------------------------------------------
    # Parameters ---------------------------------------------------------------------------------
    if PIPELINE_PARAMETERS.get("test_mode"):
        # Test mode for testing pipeline execution using low resources and iterations
        PIPELINE_PARAMETERS.update({"training_machine_cores": "16"})
        PIPELINE_PARAMETERS.update({"training_machine_ram": "104G"})
        PIPELINE_PARAMETERS.update({"training_gpu_type": "NVIDIA_TESLA_T4"})
        PIPELINE_PARAMETERS.update({"training_gpu_cores": 1})
        PIPELINE_PARAMETERS.update({"hp_machine_cores": "16"})
        PIPELINE_PARAMETERS.update({"hp_machine_ram": "104G"})
        PIPELINE_PARAMETERS.update({"hp_gpu_type": "NVIDIA_TESLA_T4"})
        PIPELINE_PARAMETERS.update({"hp_gpu_cores": 1})
        PIPELINE_PARAMETERS.update(
            {"version": "test-" + PIPELINE_PARAMETERS.get("version")}
        )
        PIPELINE_PARAMETERS.update(
            {"release": "test-" + PIPELINE_PARAMETERS.get("release")}
        )
        PIPELINE_PARAMETERS.update(
            {"pipeline_name": "test-" + PIPELINE_PARAMETERS.get("pipeline_name")}
        )
        PIPELINE_PARAMETERS.update(
            {"endpoint_name": "test-" + PIPELINE_PARAMETERS.get("endpoint_name")}
        )
        PIPELINE_PARAMETERS.update({"schedule_cron": ""})
        PIPELINE_PARAMETERS.update({"hp_training_iters_per_trial": 3})
        PIPELINE_PARAMETERS.update({"hp_ntrials": 3})
        PIPELINE_PARAMETERS.update({"training_iters": 3})

    PIPELINE_PARAMETERS.update({"region": "us-central1"})

    PIPELINE_PARAMETERS.update(
        {
            "pipeline_root": f"gs://{PIPELINE_PARAMETERS.get('pipeline_bucket_name')}/pipeline_root"  # noqa: E501
        }
    )
    PIPELINE_LABELS = {
        "application": PIPELINE_PARAMETERS.get("application_name"),
        "pipeline": PIPELINE_PARAMETERS.get("pipeline_name"),
        "model": PIPELINE_PARAMETERS.get("model_name"),
        "release": PIPELINE_PARAMETERS.get("release"),
        "version": PIPELINE_PARAMETERS.get("version"),
    }
    PIPELINE_PARAMETERS.update({"pipeline_type": "microservices"})

    print(PIPELINE_PARAMETERS)

    aip.init(
        project=PIPELINE_PARAMETERS.get("project_id"),
        staging_bucket=PIPELINE_PARAMETERS.get("pipeline_root"),
    )

    # Create Pipeline ----------------------------------------------------------------------------
    @kfp.v2.dsl.pipeline(
        name=PIPELINE_PARAMETERS.get("pipeline_name"),
        pipeline_root=PIPELINE_PARAMETERS.get("pipeline_root"),
    )
    def pipeline(
        project_id: str,
        version: str,
        pipeline_type: str,
        application_name: str,
        model_name: str,
        pipeline_name: str,
        endpoint_name: str,
        secret_path: str,
        topic_monitoring_name: str,
        release: str,
        git_url: str,
        git_branch: str,
        git_commit: str,
        test_set_pct: float,
        validation_set_pct: float,
        input_dataset_bq: str,
        output_dataset_bq: str,
        metadata_dataset_bq: str,
        metadata_table_bq: str,
        hp_training_iters_per_trial: int,
        hp_ntrials: int,
        hp_min_range_values: List[float],
        hp_max_range_values: List[float],
        hp_names: List[str],
        hp_init_values: List[float],
        training_iters: int,
        use_gpu: bool,
        schedule_cron: str,
        input_bucket_name: str,
        model_bucket_name: str,
        output_bucket_name: str,
        pipeline_bucket_name: str,
        prediction_bucket_name: str,
        drop_hp_tuning: bool,
        drop_postprocessing: bool,
        drop_deployment: bool,
        test_mode: bool,
        service_account: str,
        container_image_preprocessing: str,
        container_image_training: str,
        container_image_postprocessing: str,
        container_image_deployment: str,
        preprocessing_data_quailty_decision_rule_name: str,
        preprocessing_data_quailty_decision_rule_value: float,
        preprocessing_data_quailty_decision_rule_direction: str,
        training_model_quailty_decision_rule_name: str,
        training_model_quailty_decision_rule_value: float,
        training_model_quailty_decision_rule_direction: str,
        postprocessing_data_quailty_decision_rule_name: str,
        postprocessing_data_quailty_decision_rule_value: float,
        postprocessing_data_quailty_decision_rule_direction: str,
        region: str,
        pipeline_labels: dict,
        endpoint_machine_type: str,
        endpoint_min_replica_count: int,
        endpoint_max_replica_count: int,
        pipeline_network: str,
        endpoint_network: str,
    ):
        """
        Pipeline execution and component orchestration.

        """
        pipeline_id = dsl.PIPELINE_JOB_ID_PLACEHOLDER
        pipeline_resource = dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER

        monitoring_op = monitoring_trigger(
            project_id,
            topic_monitoring_name,
            pipeline_id,
            pipeline_name,
            pipeline_resource,
            pipeline_type,
            pipeline_labels,
            application_name,
            model_name,
            endpoint_name,
            region,
            secret_path,
            input_bucket_name,
            model_bucket_name,
            output_bucket_name,
            pipeline_bucket_name,
            prediction_bucket_name,
            test_set_pct,
            validation_set_pct,
            input_dataset_bq,
            output_dataset_bq,
            hp_training_iters_per_trial,
            hp_ntrials,
            training_iters,
            use_gpu,
            schedule_cron,
            service_account,
            container_image_preprocessing,
            container_image_training,
            container_image_postprocessing,
            container_image_deployment,
            metadata_dataset_bq,
            metadata_table_bq,
            version,
            release,
            git_url,
            git_branch,
            git_commit,
            preprocessing_data_quailty_decision_rule_name,
            preprocessing_data_quailty_decision_rule_value,
            preprocessing_data_quailty_decision_rule_direction,
            training_model_quailty_decision_rule_name,
            training_model_quailty_decision_rule_value,
            training_model_quailty_decision_rule_direction,
            postprocessing_data_quailty_decision_rule_name,
            postprocessing_data_quailty_decision_rule_value,
            postprocessing_data_quailty_decision_rule_direction,
            drop_hp_tuning,
            drop_postprocessing,
            drop_deployment,
            test_mode,
            pipeline_network,
            endpoint_network,
        )

        with dsl.ExitHandler(monitoring_op):
            preprocess_op = preprocessing(
                version,
                model_name,
                secret_path,
                test_set_pct,
                validation_set_pct,
                project_id,
                input_dataset_bq,
                input_bucket_name,
                test_mode,
                pipeline_labels,
            ).set_display_name("preprocess_op")

            with dsl.ParallelFor(
                preprocess_op.outputs["input_table_bq_uri_list"]
            ) as preprocessing_table:
                preprocess_stats_op = preprocessing_stats_validation(
                    preprocessing_data_quailty_decision_rule_name,
                    preprocessing_data_quailty_decision_rule_value,
                    preprocessing_data_quailty_decision_rule_direction,
                    preprocessing_table,
                ).set_display_name("preprocess_stats_op")

            hp_tune_op = (
                hp_tuning(
                    version,
                    model_name,
                    hp_training_iters_per_trial,
                    hp_ntrials,
                    hp_min_range_values,
                    hp_max_range_values,
                    hp_names,
                    hp_init_values,
                    use_gpu,
                    model_bucket_name,
                    drop_hp_tuning,
                    preprocess_op.outputs["input_table_bq_uri_list"],
                )
                .set_cpu_limit(PIPELINE_PARAMETERS.get("hp_machine_cores"))
                .set_memory_limit(PIPELINE_PARAMETERS.get("hp_machine_ram"))
                .add_node_selector_constraint(
                    "cloud.google.com/gke-accelerator",
                    PIPELINE_PARAMETERS.get("hp_gpu_type"),
                )
                .set_gpu_limit(PIPELINE_PARAMETERS.get("hp_gpu_cores"))
            ).set_display_name("hyper_tune_op")

            train_op = (
                training(
                    version,
                    model_name,
                    training_iters,
                    use_gpu,
                    model_bucket_name,
                    training_model_quailty_decision_rule_name,
                    training_model_quailty_decision_rule_value,
                    training_model_quailty_decision_rule_direction,
                    preprocess_op.outputs["input_table_bq_uri_list"],
                    hp_tune_op.output,
                )
                .set_cpu_limit(PIPELINE_PARAMETERS.get("training_machine_cores"))
                .set_memory_limit(PIPELINE_PARAMETERS.get("training_machine_ram"))
                .add_node_selector_constraint(
                    "cloud.google.com/gke-accelerator",
                    PIPELINE_PARAMETERS.get("training_gpu_type"),
                )
                .set_gpu_limit(PIPELINE_PARAMETERS.get("training_gpu_cores"))
            ).set_display_name("train_op")

            hp_tune_op.after(preprocess_stats_op)
            train_op.after(preprocess_stats_op)

            if not PIPELINE_PARAMETERS.get("drop_postprocessing"):
                postprocess_op = (
                    postprocessing(
                        version,
                        model_name,
                        project_id,
                        output_dataset_bq,
                        output_bucket_name,
                        test_mode,
                        secret_path,
                        pipeline_labels,
                        train_op.outputs["model_uri"],
                    )
                    .set_cpu_limit(PIPELINE_PARAMETERS.get("training_machine_cores"))
                    .set_memory_limit(PIPELINE_PARAMETERS.get("training_machine_ram"))
                    .add_node_selector_constraint(
                        "cloud.google.com/gke-accelerator",
                        PIPELINE_PARAMETERS.get("training_gpu_type"),
                    )
                    .set_gpu_limit(PIPELINE_PARAMETERS.get("training_gpu_cores"))
                ).set_display_name("postprocess_op")

                with dsl.ParallelFor(
                    postprocess_op.outputs["output_table_bq_uri_list"]
                ) as postprocessing_table:
                    postprocess_stats_op = postprocessing_stats_validation(
                        postprocessing_data_quailty_decision_rule_name,
                        postprocessing_data_quailty_decision_rule_value,
                        postprocessing_data_quailty_decision_rule_direction,
                        postprocessing_table,
                    ).set_display_name("postprocess_stats_op")

            if not PIPELINE_PARAMETERS.get("drop_deployment"):
                model_upload_op = gcc_aip.ModelUploadOp(
                    project=project_id,
                    display_name="model_upload_{}".format(
                        PIPELINE_PARAMETERS.get("model_name")
                    ),
                    #artifact_uri=train_op.outputs["model_uri"],
                    serving_container_image_uri=container_image_deployment,
                    serving_container_predict_route="/predict",
                    serving_container_health_route="/health",
                    serving_container_ports=[{"containerPort": 8080}],
                    labels=pipeline_labels,
                    serving_container_environment_variables=[
                        {
                            "name": "SECRET_PATH",
                            "value": PIPELINE_PARAMETERS.get("secret_path"),
                        },
                        {
                            "name": "PREDICTION_BUCKET_NAME",
                            "value": PIPELINE_PARAMETERS.get("prediction_bucket_name"),
                        },
                    ],
                ).set_display_name("model_upload_op")

                deployment_op = pre_deployment(
                    endpoint_name,
                    project_id,
                    region,
                    pipeline_bucket_name,
                    pipeline_labels,
                    endpoint_network,
                ).set_display_name("pre_deploy_op")

                deploy_op = gcc_aip.ModelDeployOp(
                    model=model_upload_op.outputs["model"],
                    endpoint=deployment_op.outputs["vertex_endpoint"],
                    # traffic_split=model_info['traffic'],
                    service_account=service_account,
                    dedicated_resources_machine_type=endpoint_machine_type,
                    dedicated_resources_min_replica_count=endpoint_min_replica_count,
                    dedicated_resources_max_replica_count=endpoint_max_replica_count,
                ).set_display_name("deploy_op")

                post_deployment_op = post_deployment(
                    endpoint_name,
                    project_id,
                    region,
                    endpoint_network,
                    pipeline_bucket_name,
                ).set_display_name("post_deploy_op")

                deployment_op.after(model_upload_op)
                post_deployment_op.after(deploy_op)

                if not PIPELINE_PARAMETERS.get("drop_postprocessing"):
                    model_upload_op.after(postprocess_stats_op)

    # Compiler -----------------------------------------------------------------------------------
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=f"{PIPELINE_PARAMETERS.get('pipeline_name')}.json",
    )

    # Runs or schedules a Pipeline Job -----------------------------------------------------------
    deploy_pipeline(
        compiled_pipeline=f"{PIPELINE_PARAMETERS.get('pipeline_name')}.json",
        enable_caching=PIPELINE_PARAMETERS.get("use_cache"),
        params_dict={
            "project_id": PIPELINE_PARAMETERS.get("project_id"),
            "version": PIPELINE_PARAMETERS.get("version"),
            "pipeline_type": PIPELINE_PARAMETERS.get("pipeline_type"),
            "application_name": PIPELINE_PARAMETERS.get("application_name"),
            "model_name": PIPELINE_PARAMETERS.get("model_name"),
            "pipeline_name": PIPELINE_PARAMETERS.get("pipeline_name"),
            "endpoint_name": PIPELINE_PARAMETERS.get("endpoint_name"),
            "secret_path": PIPELINE_PARAMETERS.get("secret_path"),
            "topic_monitoring_name": PIPELINE_PARAMETERS.get("topic_monitoring_name"),
            "release": PIPELINE_PARAMETERS.get("release"),
            "git_url": PIPELINE_PARAMETERS.get("git_url"),
            "git_branch": PIPELINE_PARAMETERS.get("git_branch"),
            "git_commit": PIPELINE_PARAMETERS.get("git_commit"),
            "test_set_pct": PIPELINE_PARAMETERS.get("test_set_pct"),
            "validation_set_pct": PIPELINE_PARAMETERS.get("validation_set_pct"),
            "input_dataset_bq": PIPELINE_PARAMETERS.get("input_dataset_bq"),
            "output_dataset_bq": PIPELINE_PARAMETERS.get("output_dataset_bq"),
            "metadata_dataset_bq": PIPELINE_PARAMETERS.get("metadata_dataset_bq"),
            "metadata_table_bq": PIPELINE_PARAMETERS.get("metadata_table_bq"),
            "hp_training_iters_per_trial": PIPELINE_PARAMETERS.get(
                "hp_training_iters_per_trial"
            ),
            "hp_ntrials": PIPELINE_PARAMETERS.get("hp_ntrials"),
            "hp_min_range_values": PIPELINE_PARAMETERS.get("hp_min_range_values"),
            "hp_max_range_values": PIPELINE_PARAMETERS.get("hp_max_range_values"),
            "hp_names": PIPELINE_PARAMETERS.get("hp_names"),
            "hp_init_values": PIPELINE_PARAMETERS.get("hp_init_values"),
            "training_iters": PIPELINE_PARAMETERS.get("training_iters"),
            "use_gpu": PIPELINE_PARAMETERS.get("use_gpu"),
            "schedule_cron": PIPELINE_PARAMETERS.get("schedule_cron"),
            "input_bucket_name": PIPELINE_PARAMETERS.get("input_bucket_name"),
            "model_bucket_name": PIPELINE_PARAMETERS.get("model_bucket_name"),
            "output_bucket_name": PIPELINE_PARAMETERS.get("output_bucket_name"),
            "pipeline_bucket_name": PIPELINE_PARAMETERS.get("pipeline_bucket_name"),
            "prediction_bucket_name": PIPELINE_PARAMETERS.get("prediction_bucket_name"),
            "drop_hp_tuning": PIPELINE_PARAMETERS.get("drop_hp_tuning"),
            "drop_postprocessing": PIPELINE_PARAMETERS.get("drop_postprocessing"),
            "drop_deployment": PIPELINE_PARAMETERS.get("drop_deployment"),
            "test_mode": PIPELINE_PARAMETERS.get("test_mode"),
            "service_account": PIPELINE_PARAMETERS.get("service_account"),
            "container_image_preprocessing": PIPELINE_PARAMETERS.get(
                "container_image_preprocessing"
            ),
            "container_image_training": PIPELINE_PARAMETERS.get(
                "container_image_training"
            ),
            "container_image_postprocessing": PIPELINE_PARAMETERS.get(
                "container_image_postprocessing"
            ),
            "container_image_deployment": PIPELINE_PARAMETERS.get(
                "container_image_deployment"
            ),
            "preprocessing_data_quailty_decision_rule_name": (
                PIPELINE_PARAMETERS.get("preprocessing_data_quailty_decision_rule_name")
            ),
            "preprocessing_data_quailty_decision_rule_value": (
                PIPELINE_PARAMETERS.get(
                    "preprocessing_data_quailty_decision_rule_value"
                )
            ),
            "preprocessing_data_quailty_decision_rule_direction": (
                PIPELINE_PARAMETERS.get(
                    "preprocessing_data_quailty_decision_rule_direction"
                )
            ),
            "training_model_quailty_decision_rule_name": (
                PIPELINE_PARAMETERS.get("training_model_quailty_decision_rule_name")
            ),
            "training_model_quailty_decision_rule_value": (
                PIPELINE_PARAMETERS.get("training_model_quailty_decision_rule_value")
            ),
            "training_model_quailty_decision_rule_direction": (
                PIPELINE_PARAMETERS.get(
                    "training_model_quailty_decision_rule_direction"
                )
            ),
            "postprocessing_data_quailty_decision_rule_name": (
                PIPELINE_PARAMETERS.get(
                    "postprocessing_data_quailty_decision_rule_name"
                )
            ),
            "postprocessing_data_quailty_decision_rule_value": (
                PIPELINE_PARAMETERS.get(
                    "postprocessing_data_quailty_decision_rule_value"
                )
            ),
            "postprocessing_data_quailty_decision_rule_direction": (
                PIPELINE_PARAMETERS.get(
                    "postprocessing_data_quailty_decision_rule_direction"
                )
            ),
            "region": PIPELINE_PARAMETERS.get("region"),
            "pipeline_labels": PIPELINE_LABELS,
            "endpoint_machine_type": PIPELINE_PARAMETERS.get("endpoint_machine_type"),
            "endpoint_min_replica_count": PIPELINE_PARAMETERS.get(
                "endpoint_min_replica_count"
            ),
            "endpoint_max_replica_count": PIPELINE_PARAMETERS.get(
                "endpoint_max_replica_count"
            ),
            "pipeline_network": PIPELINE_PARAMETERS.get("pipeline_network"),
            "endpoint_network": PIPELINE_PARAMETERS.get("endpoint_network"),
        },
        service_account=PIPELINE_PARAMETERS.get("service_account"),
        project_id=PIPELINE_PARAMETERS.get("project_id"),
        pipeline_root=PIPELINE_PARAMETERS.get("pipeline_root"),
        schedule=PIPELINE_PARAMETERS.get("schedule_cron"),
        pipeline_labels=PIPELINE_LABELS,
        network=PIPELINE_PARAMETERS.get("pipeline_network"),
    )
