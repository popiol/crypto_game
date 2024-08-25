import glob

from src.environment import Environment


class TestReports:

    def test_reports(self):
        environment = Environment("config/config.yml")
        reports = environment.reports
        model_registry = environment.model_registry
        model_registry.download_aggregated_metrics(reports.aggregated_path)
        files = glob.glob(reports.aggregated_path + "/*.json")
        reports.copy_custom_metrics(files)
        df = reports.calc_change_in_time(files)
        df.to_csv(reports.change_in_time_path, index=False)

    def test_calc_leader_stats(self):
        environment = Environment("config/config.yml")
        reports = environment.reports
        df = reports.calc_leader_stats()
        print(df)
        df.to_csv(reports.leader_stats_path, index=False)
