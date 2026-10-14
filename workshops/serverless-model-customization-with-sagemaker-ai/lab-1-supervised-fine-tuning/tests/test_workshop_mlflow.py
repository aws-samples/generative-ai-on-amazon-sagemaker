"""Offline tests: python -m unittest discover -s tests -v from the workshop root."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workshop_mlflow import resolve_app

ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/test"
ROLE = "arn:aws:iam::123456789012:role/workshop"
STORE = "s3://workshop-artifacts/mlflow-artifacts"
NAME = "nemotron-contractnli-workshop"
STACK_UUID = "4097f350-c7c9-11f1-a750-0e05a95aba6d"
CFN_NAME = f"workshop-mlflow-{STACK_UUID}"
CFN_ARN = ARN.rsplit("/", 1)[0] + "/" + CFN_NAME
STACK_ARN = f"arn:aws:cloudformation:us-west-2:123456789012:stack/workshop/{STACK_UUID}"


class MlflowSetupTests(unittest.TestCase):
    def setUp(self):
        self.sm = Mock(meta=types.SimpleNamespace(region_name="us-west-2"))
        self.sm.describe_mlflow_app.return_value = {"Status": "Created"}
        self.sm.create_mlflow_app.return_value = {"Arn": ARN}
        self.app_pages, self.tag_pages = Mock(), Mock()
        self.sm.get_paginator.side_effect = {
            "list_mlflow_apps": self.app_pages, "list_tags": self.tag_pages,
        }.__getitem__
        self.tag_pages.paginate.return_value = [{"Tags": [
            {"Key": "WorkshopStackId", "Value": STACK_ARN}]}]
        self.pages([])
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.sleep = self.enterContext(patch("workshop_mlflow.time.sleep"))

    def pages(self, apps):
        self.app_pages.paginate.return_value = [
            {"Summaries": [{"Name": "other", "Arn": ARN}]}, {"Summaries": apps}
        ]

    def native(self, result=ARN):
        resolver = Mock(return_value=result)
        session_class = Mock()
        modules = {
            "sagemaker.core.helper.session_helper": types.SimpleNamespace(Session=session_class),
            "sagemaker.train.common_utils.finetune_utils": types.SimpleNamespace(
                _resolve_mlflow_resource_arn=resolver),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        return resolver, session_class

    def test_explicit_ready_does_not_discover_or_create(self):
        for status in ("Created", "Updated"):
            self.sm.describe_mlflow_app.return_value = {"Status": status}
            self.assertEqual(resolve_app(self.sm, ARN), ARN)
        self.sm.get_paginator.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()
        self.sm.update_mlflow_app.assert_not_called()

    def test_bad_explicit_arn_never_falls_back(self):
        for arn in ("invalid", ARN.replace("us-west-2", "us-east-1"),
                    ARN.replace("mlflow-app", "mlflow-tracking-server")):
            with self.assertRaises(ValueError):
                resolve_app(self.sm, arn, native=True)
        self.sm.get_paginator.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()

    def test_explicit_missing_or_inaccessible_never_creates(self):
        for code in ("ResourceNotFound", "AccessDeniedException"):
            error = ClientError({"Error": {"Code": code, "Message": "test"}}, "DescribeMlflowApp")
            self.sm.describe_mlflow_app.side_effect = error
            with self.assertRaises(ClientError) as raised:
                resolve_app(self.sm, ARN)
            self.assertIs(raised.exception, error)
        self.sm.create_mlflow_app.assert_not_called()

    def test_missing_arn_reuses_named_app_across_pages_and_waits(self):
        self.pages([{"Name": NAME, "Arn": ARN}])
        self.sm.describe_mlflow_app.side_effect = [{"Status": "Creating"}, {"Status": "Updated"}]
        self.assertEqual(resolve_app(self.sm), ARN)
        self.sm.get_paginator.assert_called_once_with("list_mlflow_apps")
        self.sm.create_mlflow_app.assert_not_called()
        self.sleep.assert_called_once()

    def test_missing_arn_creates_then_waits(self):
        self.sm.describe_mlflow_app.side_effect = [{"Status": "Creating"}, {"Status": "Created"}]
        self.assertEqual(resolve_app(self.sm, role_arn=ROLE, artifact_store_uri=STORE), ARN)
        self.sm.create_mlflow_app.assert_called_once_with(
            Name=NAME, RoleArn=ROLE, ArtifactStoreUri=STORE, AccountDefaultStatus="DISABLED")
        self.sm.update_mlflow_app.assert_not_called()

    def test_failed_or_deleted_apps_stop_setup(self):
        for status in ("CreateFailed", "UpdateFailed", "Deleting", "DeleteFailed", "Deleted"):
            self.sm.describe_mlflow_app.return_value = {"Status": status}
            with self.assertRaisesRegex(RuntimeError, status):
                resolve_app(self.sm, ARN)
        self.sm.create_mlflow_app.assert_not_called()

    def test_created_app_failure_stops_setup(self):
        self.sm.describe_mlflow_app.side_effect = [{"Status": "Creating"}, {"Status": "CreateFailed"}]
        with self.assertRaisesRegex(RuntimeError, "CreateFailed"):
            resolve_app(self.sm, role_arn=ROLE, artifact_store_uri=STORE)
        self.assertEqual(self.sm.create_mlflow_app.call_count, 1)

    def test_timeout_does_not_create_replacement(self):
        self.sm.describe_mlflow_app.return_value = {"Status": "Updating"}
        with patch("workshop_mlflow.time.monotonic", side_effect=[0, 0, 11]):
            with self.assertRaisesRegex(TimeoutError, "still Updating"):
                resolve_app(self.sm, ARN, timeout_seconds=10, poll_seconds=2)
        self.sm.create_mlflow_app.assert_not_called()

    def test_permission_and_quota_errors_propagate(self):
        for operation, code in (("ListMlflowApps", "AccessDeniedException"),
                                ("CreateMlflowApp", "ResourceLimitExceeded")):
            error = ClientError({"Error": {"Code": code, "Message": "test"}}, operation)
            method = (self.app_pages.paginate if operation == "ListMlflowApps"
                      else self.sm.create_mlflow_app)
            method.side_effect = error
            with self.assertRaises(ClientError) as raised:
                resolve_app(self.sm, role_arn=ROLE, artifact_store_uri=STORE)
            self.assertIs(raised.exception, error)
            method.side_effect = None

    def test_ambiguous_name_and_incomplete_creation_settings(self):
        self.pages([{"Name": NAME, "Arn": ARN}] * 2)
        with self.assertRaisesRegex(RuntimeError, "Multiple"):
            resolve_app(self.sm)
        self.pages([])
        with self.assertRaisesRegex(ValueError, "both a role ARN"):
            resolve_app(self.sm, role_arn=ROLE)
        self.sm.create_mlflow_app.assert_not_called()

    def test_native_discovery_or_creation_is_delegated_without_upgrade(self):
        resolver, session_class = self.native()
        self.sm.describe_mlflow_app.side_effect = [{"Status": "Creating"}, {"Status": "Created"}]
        session = object()
        self.assertEqual(resolve_app(self.sm, native=True, session=session), ARN)
        resolver.assert_called_once_with(session)
        session_class.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()

    def test_native_logged_failure_never_starts_untracked_training(self):
        resolver, _ = self.native(None)
        with self.assertRaisesRegex(RuntimeError, "Training was not started"):
            resolve_app(self.sm, native=True)
        resolver.assert_called_once()
        self.sm.describe_mlflow_app.assert_not_called()

    def test_tagged_stack_app_wins_over_named_duplicate(self):
        import tempfile
        resolver, _ = self.native()
        spec = importlib.util.spec_from_file_location("lab_config", ROOT / "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        config.MLFLOW_APP_NAME = NAME
        with tempfile.TemporaryDirectory() as directory:
            with patch("pathlib.Path.home", return_value=Path(directory)):
                for duplicate in ([], [{"Name": NAME, "Arn": ARN}]):
                    self.pages(duplicate + [{"Name": CFN_NAME, "Arn": CFN_ARN}])
                    self.sm.describe_mlflow_app.return_value = {
                        "Status": "Created", "RoleArn": ROLE,
                        "ArtifactStoreUri": "s3://stack-bucket/mlflow-artifacts/",
                    }
                    self.assertEqual(config.resolve_mlflow_app(self.sm), CFN_ARN)
        resolver.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()
        self.sm.update_mlflow_app.assert_not_called()
        self.tag_pages.paginate.assert_called_with(ResourceArn=CFN_ARN)

    def test_explicit_arn_wins_even_with_cfn_and_duplicate(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}, {"Name": NAME, "Arn": ARN}])
        self.assertEqual(resolve_app(self.sm, ARN, native=True), ARN)
        self.sm.get_paginator.assert_not_called()

    def test_cfn_tags_are_paginated(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        self.tag_pages.paginate.return_value = [
            {"Tags": [{"Key": "Other", "Value": "tag"}], "NextToken": "next"},
            {"Tags": [{"Key": "WorkshopStackId", "Value": STACK_ARN}]},
        ]
        self.assertEqual(resolve_app(self.sm), CFN_ARN)
        self.sm.create_mlflow_app.assert_not_called()

    def test_prefix_or_unrelated_tag_is_not_enough(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        for tags in ([], [{"Key": "Purpose", "Value": "workshop"}],
                     [{"Key": "WorkshopStackId", "Value": "not-a-stack"}],
                     [{"Key": "WorkshopStackId", "Value": STACK_ARN.replace(STACK_UUID, "00000000-0000-0000-0000-000000000000")}],
                     [{"Key": "WorkshopStackId", "Value": STACK_ARN.replace("123456789012", "999999999999")}]):
            with self.subTest(tags=tags):
                self.tag_pages.paginate.return_value = [{"Tags": tags}]
                self.assertEqual(resolve_app(self.sm, role_arn=ROLE, artifact_store_uri=STORE), ARN)
        self.sm.describe_mlflow_app.assert_called_with(Arn=ARN)
        self.sm.update_mlflow_app.assert_not_called()

    def test_unrelated_name_does_not_request_tags(self):
        self.assertEqual(resolve_app(self.sm, role_arn=ROLE, artifact_store_uri=STORE), ARN)
        self.tag_pages.paginate.assert_not_called()

    def test_role_scope_and_artifact_scope(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        self.sm.describe_mlflow_app.return_value = {
            "Status": "Created", "RoleArn": ROLE, "ArtifactStoreUri": STORE + "/"}
        self.assertEqual(resolve_app(self.sm, role_arn=ROLE), CFN_ARN)
        self.assertEqual(resolve_app(self.sm, artifact_store_uri=STORE), CFN_ARN)
        self.sm.create_mlflow_app.assert_not_called()
        # Mismatched scopes must not select another workshop's app.
        self.assertEqual(resolve_app(self.sm, role_arn=ROLE + "-other", artifact_store_uri=STORE), ARN)
        with self.assertRaisesRegex(ValueError, "both a role ARN"):
            resolve_app(self.sm, artifact_store_uri="s3://other/mlflow-artifacts")

    def test_multiple_cfn_apps_stop_even_with_named_duplicate(self):
        other_uuid = "00000000-0000-0000-0000-000000000000"
        self.pages([{"Name": NAME, "Arn": ARN}, {"Name": CFN_NAME, "Arn": CFN_ARN},
                    {"Name": CFN_NAME.replace(STACK_UUID, other_uuid),
                     "Arn": CFN_ARN.replace(STACK_UUID, other_uuid)}])
        self.tag_pages.paginate.side_effect = lambda ResourceArn: [{"Tags": [{
            "Key": "WorkshopStackId", "Value": STACK_ARN if ResourceArn == CFN_ARN
            else STACK_ARN.replace(STACK_UUID, other_uuid)}]}]
        resolver, _ = self.native()
        for native in (False, True):
            with self.assertRaisesRegex(RuntimeError, "Multiple.*MLFLOW_APP_ARN"):
                resolve_app(self.sm, native=native)
        self.sm.create_mlflow_app.assert_not_called()
        resolver.assert_not_called()

    def test_pending_cfn_waits_instead_of_creating(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        resolver, _ = self.native()
        for native in (False, True):
            self.sm.describe_mlflow_app.side_effect = [{"Status": "Creating"}, {"Status": "Updated"}]
            self.assertEqual(resolve_app(self.sm, native=native), CFN_ARN)
        self.assertEqual(self.sleep.call_count, 2)
        self.sm.create_mlflow_app.assert_not_called()
        resolver.assert_not_called()

    def test_failed_cfn_never_falls_back_to_duplicate_or_native(self):
        self.pages([{"Name": NAME, "Arn": ARN}, {"Name": CFN_NAME, "Arn": CFN_ARN}])
        resolver, _ = self.native()
        for status in ("CreateFailed", "UpdateFailed", "Deleting", "DeleteFailed", "Deleted", "Unknown"):
            for native in (False, True):
                self.sm.describe_mlflow_app.return_value = {"Status": status}
                with self.assertRaisesRegex(RuntimeError, status):
                    resolve_app(self.sm, native=native)
        self.sm.create_mlflow_app.assert_not_called()
        resolver.assert_not_called()

    def test_cfn_discovery_access_denied_never_means_absent(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        resolver, _ = self.native()
        for method in (self.app_pages.paginate, self.tag_pages.paginate, self.sm.describe_mlflow_app):
            error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "test"}}, "Discovery")
            method.side_effect = error
            for native in (False, True):
                with self.assertRaises(ClientError) as raised:
                    resolve_app(self.sm, native=native, role_arn=ROLE, artifact_store_uri=STORE)
                self.assertIs(raised.exception, error)
            method.side_effect = None
        self.sm.create_mlflow_app.assert_not_called()
        resolver.assert_not_called()

    def test_native_discovery_denied_before_sdk_fallback(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        resolver, _ = self.native()
        error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "test"}}, "ListTags")
        self.tag_pages.paginate.side_effect = error
        with self.assertRaises(ClientError):
            resolve_app(self.sm, native=True)
        resolver.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()

    def test_cfn_timeout_never_falls_back(self):
        self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
        resolver, _ = self.native()
        self.sm.describe_mlflow_app.return_value = {"Status": "Creating"}
        with patch("workshop_mlflow.time.monotonic", side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                resolve_app(self.sm, native=True, timeout_seconds=10)
        resolver.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()

    def test_real_botocore_pagination_and_tag_schema_offline(self):
        from botocore.session import Session
        from botocore.stub import Stubber

        client = Session().create_client("sagemaker", region_name="us-west-2",
                                         aws_access_key_id="testing", aws_secret_access_key="testing")
        with Stubber(client) as stub:
            stub.add_response("list_mlflow_apps", {"Summaries": [{"Name": NAME, "Arn": ARN}],
                                                   "NextToken": "apps-page-2"}, {})
            stub.add_response("list_mlflow_apps", {"Summaries": [{"Name": CFN_NAME, "Arn": CFN_ARN}]},
                              {"NextToken": "apps-page-2"})
            stub.add_response("list_tags", {"Tags": [], "NextToken": "tags-page-2"}, {"ResourceArn": CFN_ARN})
            stub.add_response("list_tags", {"Tags": [{"Key": "WorkshopStackId", "Value": STACK_ARN}]},
                              {"ResourceArn": CFN_ARN, "NextToken": "tags-page-2"})
            stub.add_response("describe_mlflow_app", {"Status": "Created", "RoleArn": ROLE,
                                                     "ArtifactStoreUri": STORE + "/"}, {"Arn": CFN_ARN})
            stub.add_response("describe_mlflow_app", {"Status": "Created"}, {"Arn": CFN_ARN})
            self.assertEqual(resolve_app(client, role_arn=ROLE, artifact_store_uri=STORE), CFN_ARN)
            stub.assert_no_pending_responses()
        client.close()

    def test_incomplete_pagination_stops_before_fallback(self):
        resolver, _ = self.native()
        for operation in ("apps", "tags"):
            self.pages([{"Name": CFN_NAME, "Arn": CFN_ARN}])
            if operation == "apps":
                self.app_pages.paginate.return_value = [{"Summaries": [], "NextToken": "more"}] * 100
            else:
                self.tag_pages.paginate.return_value = [{"Tags": [], "NextToken": "more"}] * 100
            with self.assertRaisesRegex(RuntimeError, "100 pages.*MLFLOW_APP_ARN"):
                resolve_app(self.sm, native=True)
        self.sm.create_mlflow_app.assert_not_called()
        resolver.assert_not_called()

    def test_lab_config_explicit_arn_never_reads_home_metadata(self):
        spec = importlib.util.spec_from_file_location("lab_config", ROOT / "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        config.MLFLOW_APP_ARN = ARN
        with patch("pathlib.Path.home", side_effect=AssertionError("No home-file handoff")):
            self.assertEqual(config.resolve_mlflow_app(self.sm), ARN)
        self.sm.get_paginator.assert_not_called()
        self.sm.create_mlflow_app.assert_not_called()

    def test_lab_config_discovers_before_native_fallback(self):
        spec = importlib.util.spec_from_file_location("lab_config", ROOT / "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        resolver, _ = self.native()
        with patch("pathlib.Path.home", side_effect=AssertionError("No home-file handoff")):
            self.assertEqual(config.resolve_mlflow_app(self.sm), ARN)
        resolver.assert_called_once()
        self.sm.get_paginator.assert_called()
        self.sm.create_mlflow_app.assert_not_called()


if __name__ == "__main__":
    unittest.main()
