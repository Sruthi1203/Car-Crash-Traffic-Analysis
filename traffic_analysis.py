import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd

class TrafficAnalyzer:
    def __init__(self):
        try:
            self.spark = SparkSession.builder \
                .appName("TrafficAnalysis") \
                .config("spark.mongodb.input.uri", "mongodb://localhost:27017/TrafficData.csv") \
                .config("spark.mongodb.output.uri", "mongodb://localhost:27017/TrafficAnalysis") \
                .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
                .master("local[*]") \
                .getOrCreate()

            # Create directories for outputs
            self.output_dir = 'analysis_results'
            self.viz_dir = os.path.join(self.output_dir, 'visualizations3')
            self.powerbi_dir = os.path.join(self.output_dir, 'powerbi_data1')
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
            os.makedirs(self.powerbi_dir, exist_ok=True)

            sns.set(style="whitegrid")
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Traffic Analyzer initialized successfully")
            
            self.anomaly_results = None
            self.anomaly_threshold = None
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def load_data_to_spark(self):
        try:
            csv_df = self.spark.read.format("mongo").load()
            total_records = csv_df.count()
            self.logger.info(f"Total records loaded: {total_records}")
            csv_df = csv_df.withColumn("sequence_id", F.monotonically_increasing_id())
            self.logger.info(f"Loaded {csv_df.count()} CSV records")
            return csv_df
        except Exception as e:
            self.logger.error(f"Data loading error: {str(e)}")
            raise

    def transform_to_frame_level(self, csv_df):
        try:
            frame_cols = [f"frame_{i}" for i in range(1, 51)]
            frame_array = F.array(*[F.col(c) for c in frame_cols])
            frame_df = csv_df.select(
                'sequence_id',
                'timing',
                'egoinvolve',
                'weather',
                'vidname',
                'startframe',
                F.posexplode(frame_array).alias('frame_number', 'crash_detected')
            )
            
            frame_df = frame_df.withColumn(
                'time_of_day',
                F.when(F.lower(F.col('timing')).contains("day"), "Day")
                 .otherwise("Night")
            )
            
            total_frames = frame_df.count()
            self.logger.info(f"Total frames after transformation: {total_frames}")
            return frame_df
        except Exception as e:
            self.logger.error(f"Frame transformation error: {str(e)}")
            raise

    def analyze_crash_patterns(self, df):
        try:
            crash_stats = df.agg(
                F.sum('crash_detected').alias('total_crashes'),
                F.avg('crash_detected').alias('crash_rate'),
                F.count('*').alias('total_frames')
            ).collect()[0]
            
            metrics = {
                'total_crashes': int(crash_stats['total_crashes']),
                'crash_rate': float(crash_stats['crash_rate']),
                'total_frames': int(crash_stats['total_frames']),
                'crash_percentage': float(crash_stats['crash_rate'] * 100)
            }
            
            self.logger.info(f"Crash analysis metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Crash pattern analysis error: {str(e)}")
            raise

    def detect_anomalies(self, df):
        try:
            # Calculate crashes per sequence with additional metrics
            crashes_per_sequence = df.groupBy('sequence_id', 'time_of_day', 'weather', 'egoinvolve') \
                               .agg(
                                   F.sum('crash_detected').alias('crash_count'),
                                   F.avg('crash_detected').alias('crash_density'),
                                   F.count('*').alias('total_frames')
                               )
            
            # Convert to pandas for percentile calculation
            crashes_pd = crashes_per_sequence.toPandas()
            
            # Calculate multiple thresholds
            stats = crashes_per_sequence.agg(
                F.avg('crash_count').alias('avg_crashes'),
                F.stddev('crash_count').alias('std_crashes')
            ).collect()[0]
            
            avg_crashes = float(stats['avg_crashes'])
            std_crashes = float(stats['std_crashes'])
            
            # Use multiple criteria for anomaly detection
            statistical_threshold = avg_crashes + std_crashes
            percentile_threshold = np.percentile(crashes_pd['crash_count'], 90)
            
            self.anomaly_threshold = min(statistical_threshold, percentile_threshold)
            
            # Mark sequences as anomalous
            crashes_pd['is_anomaly'] = crashes_pd['crash_count'] > self.anomaly_threshold
            
            self.anomaly_results = crashes_pd
            
            anomalies_count = len(crashes_pd[crashes_pd['is_anomaly']])
            total_sequences = len(crashes_pd)
            
            metrics = {
                'anomalies_count': anomalies_count,
                'total_sequences': total_sequences,
                'anomaly_percentage': float((anomalies_count / total_sequences) * 100),
                'avg_crashes_per_sequence': avg_crashes,
                'std_crashes': std_crashes,
                'statistical_threshold': statistical_threshold,
                'percentile_threshold': percentile_threshold,
                'final_threshold': self.anomaly_threshold
            }
            
            self.logger.info(f"Anomaly detection metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {str(e)}")
            raise

    def create_anomaly_visualizations(self):
        try:
            if self.anomaly_results is None or self.anomaly_threshold is None:
                self.logger.warning("No anomaly results available for visualization")
                return

            # 1. Distribution with Threshold
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.anomaly_results, x='crash_count', bins=30, color='skyblue')
            plt.axvline(x=self.anomaly_threshold, color='r', linestyle='--', 
                       label=f'Anomaly Threshold ({self.anomaly_threshold:.2f})')
            plt.title('Distribution of Crashes per Sequence\nwith Anomaly Threshold')
            plt.xlabel('Number of Crashes')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'crash_distribution_with_anomalies.png'))
            plt.close()

            anomaly_data = self.anomaly_results[self.anomaly_results['crash_count'] > self.anomaly_threshold]
            
            if len(anomaly_data) > 0:
                # Additional visualizations for anomalies
                self._create_detailed_anomaly_visualizations(anomaly_data)
            else:
                self.logger.info("No anomalies detected for detailed visualizations")

            self.logger.info("Anomaly visualizations created successfully")
        except Exception as e:
            self.logger.error(f"Anomaly visualization error: {str(e)}")
            raise

    def _create_detailed_anomaly_visualizations(self, anomaly_data):
        """Creates detailed visualizations for anomalous data"""
        try:
            # 2. Time of Day Analysis
            plt.figure(figsize=(10, 6))
            time_of_day_counts = sns.countplot(data=anomaly_data, x='time_of_day')
            plt.title('Anomalous Crash Sequences by Time of Day')
            for i in time_of_day_counts.containers:
                plt.bar_label(i)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'anomalies_by_time.png'))
            plt.close()

            # 3. Weather Analysis
            plt.figure(figsize=(12, 6))
            weather_counts = sns.countplot(data=anomaly_data, x='weather')
            plt.title('Anomalous Crash Sequences by Weather')
            plt.xticks(rotation=45)
            for i in weather_counts.containers:
                plt.bar_label(i)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'anomalies_by_weather.png'))
            plt.close()

            # 4. Ego Involvement Analysis
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.anomaly_results, x='crash_count', y='egoinvolve', 
                          hue='is_anomaly', style='is_anomaly', s=100)
            plt.axvline(x=self.anomaly_threshold, color='r', linestyle='--', 
                       label='Anomaly Threshold')
            plt.title('Crash Count vs Ego Involvement\nHighlighting Anomalies')
            plt.legend(title='Is Anomaly')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'anomalies_ego_scatter.png'))
            plt.close()

            # Additional visualizations...
            self._create_advanced_anomaly_visualizations(anomaly_data)

        except Exception as e:
            self.logger.error(f"Detailed anomaly visualization error: {str(e)}")
            raise

    def _create_advanced_anomaly_visualizations(self, anomaly_data):
        """Creates advanced visualization for anomalous data"""
        try:
            # 5. Box Plot Analysis
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.anomaly_results, x='weather', y='crash_count')
            plt.axhline(y=self.anomaly_threshold, color='r', linestyle='--', 
                       label='Anomaly Threshold')
            plt.title('Crash Distribution by Weather\nwith Anomaly Threshold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'crash_boxplot_weather_anomalies.png'))
            plt.close()

            # 6. Heatmap Analysis
            plt.figure(figsize=(12, 8))
            pivot_table = pd.crosstab(anomaly_data['weather'], anomaly_data['time_of_day'])
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Anomaly Distribution: Weather vs Time of Day')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'anomaly_feature_heatmap.png'))
            plt.close()

            # 7. Density Timeline
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=anomaly_data.sort_values('sequence_id'), 
                        x='sequence_id', y='crash_density')
            plt.title('Crash Density for Anomalous Sequences')
            plt.xlabel('Sequence ID')
            plt.ylabel('Crash Density')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'anomaly_density_timeline.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Advanced anomaly visualization error: {str(e)}")
            raise
    def export_results(self, df, metrics):
        """Export analysis results and metrics to JSON"""
        try:
            export_metrics = metrics.copy()
            export_metrics['validation_timestamp'] = str(datetime.now())
            export_metrics['total_frames'] = int(df.count())
            
            # Ensure all metrics are JSON serializable
            for key in export_metrics:
                if isinstance(export_metrics[key], (np.integer, np.floating)):
                    export_metrics[key] = float(export_metrics[key])
                elif isinstance(export_metrics[key], pyspark.sql.column.Column):
                    export_metrics[key] = str(export_metrics[key])
            
            # Export frame-level results
            df.write.mode('overwrite').json(os.path.join(self.output_dir, 'analysis_results.json'))
            
            # Export metrics
            with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                json.dump(export_metrics, f, indent=4)
            
            self.logger.info("Results exported successfully")
            self.logger.info(f"Final metrics: {export_metrics}")
        except Exception as e:
            self.logger.error(f"Export error: {str(e)}")
            raise

    def export_for_powerbi(self, df):
        """Export data in formats suitable for Power BI"""
        try:
            # 1. Frame-level data
            frame_data = df.select(
                'sequence_id', 
                'time_of_day', 
                'egoinvolve', 
                'weather',
                'frame_number',
                'crash_detected'
            ).toPandas()
            
            frame_data.to_csv(os.path.join(self.powerbi_dir, 'frame_level_data.csv'), index=False)

            # 2. Sequence-level data with anomalies
            if self.anomaly_results is not None:
                self.anomaly_results.to_csv(
                    os.path.join(self.powerbi_dir, 'sequence_level_data.csv'), 
                    index=False
                )

            # 3. Summary statistics
            crash_summary = df.groupBy('time_of_day', 'weather', 'egoinvolve') \
                .agg(
                    F.sum('crash_detected').alias('total_crashes'),
                    F.count('*').alias('total_frames'),
                    F.avg('crash_detected').alias('crash_rate')
                ).toPandas()
                
            crash_summary.to_csv(os.path.join(self.powerbi_dir, 'crash_summary.csv'), index=False)

            # Create specific time-of-day crash summary
            crashes_by_time = df.filter(F.col('crash_detected') == 1) \
                            .groupBy('time_of_day') \
                            .agg(F.count('*').alias('crash_count')) \
                            .toPandas()
            
            # Export to CSV
            crashes_by_time.to_csv(os.path.join(self.powerbi_dir, 'crashes_by_time.csv'), index=False)

            # 4. Anomaly metrics
            if self.anomaly_results is not None:
                anomaly_summary = self.anomaly_results.groupby(['time_of_day', 'weather']) \
                    .agg({
                        'is_anomaly': 'sum',
                        'crash_count': ['mean', 'max'],
                        'crash_density': ['mean', 'max']
                    }).reset_index()
                anomaly_summary.columns = ['time_of_day', 'weather', 'anomaly_count', 
                                         'avg_crashes', 'max_crashes',
                                         'avg_density', 'max_density']
                anomaly_summary.to_csv(os.path.join(self.powerbi_dir, 'anomaly_summary.csv'), index=False)

            self.logger.info("Data exported for Power BI successfully")

            # Create Power BI README
            readme_content = """
            Power BI Data Files:
            1. frame_level_data.csv: Individual frame-level crash data
            2. sequence_level_data.csv: Sequence-level data with anomaly flags
            3. crash_summary.csv: Aggregated crash statistics
            4. anomaly_summary.csv: Anomaly analysis summary

            Suggested Power BI Visualizations:
            1. Crash Distribution Dashboard:
                - Time of day crash distribution (Stacked column chart)
                - Weather impact on crashes (Pie chart)
                - Ego involvement analysis (Donut chart)

            2. Anomaly Analysis Dashboard:
                - Anomaly distribution by weather (Column chart)
                - Anomaly timeline (Line chart)
                - Anomaly density heatmap (Matrix visual)

            3. Sequence Analysis Dashboard:
                - Crash density by sequence (Line chart)
                - Anomaly detection threshold visualization (Area chart)
            4. KPI Dashboard:
                - Total crashes counter
                - Anomaly percentage gauge
                - Crash rate by condition cards
            """
            
            with open(os.path.join(self.powerbi_dir, 'README.txt'), 'w') as f:
                f.write(readme_content)

        except Exception as e:
            self.logger.error(f"Power BI export error: {str(e)}")
            raise

    def create_visualizations(self, df):
        try:
            pandas_df = df.toPandas()
            
            if 'startframe' not in pandas_df.columns:
                pandas_df['startframe'] = pandas_df['frame_number']

            # 1. Number of Crashes by Time of Day
            crashes_by_time = df.filter(F.col('crash_detected') == 1) \
                              .groupBy('time_of_day') \
                              .agg(F.count('*').alias('crash_count')) \
                              .toPandas()

            plt.figure(figsize=(10, 6))
            sns.barplot(data=crashes_by_time, x='time_of_day', y='crash_count', palette="viridis")
            plt.title('Number of Crashes by Time of Day\n(Total Frames: 75,000)')
            plt.xlabel('Time of Day')
            plt.ylabel('Number of Crashes')

            for i, count in enumerate(crashes_by_time['crash_count']):
                plt.text(i, count, str(count), ha='center', va='bottom')

            total_crashes = crashes_by_time['crash_count'].sum()
            plt.text(0.5, -0.15, 
                    f'Total Crashes: {total_crashes} out of 75,000 frames ({(total_crashes/75000*100):.2f}%)',
                    ha='center', va='center', transform=plt.gca().transAxes)

            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'crashes_by_time_of_day_corrected.png'))
            plt.close()

            # Create all other visualizations...
            self._create_standard_visualizations(pandas_df)
            
            # Create anomaly visualizations
            self.create_anomaly_visualizations()

            self.logger.info("All visualizations created successfully")
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
            raise

    def _create_standard_visualizations(self, pandas_df):
        """Creates standard visualizations for crash analysis"""
        try:
            # 2. Traffic Flow Timeline
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=pandas_df, x='frame_number', y='crash_detected', ci=None)
            plt.title('Traffic Flow - Crash Detection Over Frames')
            plt.xlabel('Frame Number')
            plt.ylabel('Crash Detected')
            plt.savefig(os.path.join(self.viz_dir, 'traffic_flow.png'))
            plt.close()

            # Continue with other standard visualizations...
            self._create_distribution_visualizations(pandas_df)
            self._create_correlation_visualizations(pandas_df)

        except Exception as e:
            self.logger.error(f"Standard visualization error: {str(e)}")
            raise

    def _create_distribution_visualizations(self, pandas_df):
        """Creates distribution-related visualizations"""
        try:
            # 3. Crash Distribution
            crash_counts = pandas_df['crash_detected'].value_counts()
            plt.figure(figsize=(6, 6))
            crash_counts.plot.pie(autopct='%1.1f%%', labels=['No Crash', 'Crash'], colors=['lightblue', 'red'])
            plt.title('Crash Distribution')
            plt.ylabel('')
            plt.savefig(os.path.join(self.viz_dir, 'crash_distribution.png'))
            plt.close()

            # 4. Heatmap of Crashes
            heatmap_data = pandas_df.pivot_table(
                index='sequence_id',
                columns='frame_number',
                values='crash_detected',
                aggfunc='sum',
                fill_value=0
            )
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Number of Crashes'})
            plt.title('Heatmap of Crashes by Sequence and Frame')
            plt.xlabel('Frame Number')
            plt.ylabel('Sequence ID')
            plt.savefig(os.path.join(self.viz_dir, 'crash_heatmap.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Distribution visualization error: {str(e)}")
            raise

    def _create_correlation_visualizations(self, pandas_df):
        """Creates correlation-related visualizations"""
        try:
            crashes_by_time_pd = pandas_df[pandas_df['crash_detected'] == 1]

            # 5. Ego Involvement vs Crashes
            plt.figure(figsize=(10, 6))
            sns.countplot(data=crashes_by_time_pd, x='egoinvolve', palette="coolwarm")
            plt.title('Crashes by Ego Involvement Level')
            plt.savefig(os.path.join(self.viz_dir, 'crashes_by_ego_involve.png'))
            plt.close()

            # 6. Weather Condition vs Crashes
            plt.figure(figsize=(10, 6))
            sns.countplot(data=crashes_by_time_pd, x='weather', palette="coolwarm")
            plt.title('Crashes by Weather Condition')
            plt.savefig(os.path.join(self.viz_dir, 'crashes_by_weather.png'))
            plt.close()

            # 7. Accidents Based on Frame Numbers
            accidents_per_frame = pandas_df.groupby('startframe')['vidname'].count().reset_index()
            plt.figure(figsize=(10, 6))
            plt.scatter(accidents_per_frame['startframe'], accidents_per_frame['vidname'], color='blue', alpha=0.5)
            plt.title('Accidents Based on Frame Numbers')
            plt.xlabel('Frame Number (startframe)')
            plt.ylabel('Number of Accidents (vidname count)')
            plt.grid(True)
            plt.savefig(os.path.join(self.viz_dir, 'accidents_based_on_frame_numbers.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Correlation visualization error: {str(e)}")
            raise

    def process_and_analyze(self):
        try:
            csv_df = self.load_data_to_spark()
            frame_df = self.transform_to_frame_level(csv_df)
            
            metrics = self.analyze_crash_patterns(frame_df)
            anomalies = self.detect_anomalies(frame_df)
            metrics.update(anomalies)
            
            # Create visualizations
            self.create_visualizations(frame_df)
            
            # Export results
            self.export_results(frame_df, metrics)
            
            # Export for Power BI
            self.export_for_powerbi(frame_df)
            
            self.logger.info("Analysis pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = TrafficAnalyzer()
    analyzer.process_and_analyze()