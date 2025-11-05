#!/usr/bin/env python3
"""
实体配置验证脚本 (Entity Configuration Validation Script)

验证YAML实体配置文件的完整性和一致性。

Usage:
    python scripts/validation/validate-entities.py --config-dir configs/content
    python scripts/validation/validate-entities.py --config-dir configs/content --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import yaml


class EntityValidator:
    """实体验证器"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # 存储已加载的实体
        self.stages: List[Dict] = []
        self.modules: List[Dict] = []
        self.projects: List[Dict] = []
        self.datasets: List[Dict] = []

    def load_yaml(self, file_path: Path) -> Dict:
        """加载YAML文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"❌ 无法加载{file_path.name}: {e}")
            return {}

    def load_all_entities(self):
        """加载所有实体配置"""
        stages_file = self.config_dir / "stages.yaml"
        modules_file = self.config_dir / "modules.yaml"
        projects_file = self.config_dir / "projects.yaml"
        datasets_file = self.config_dir / "datasets.yaml"

        if stages_file.exists():
            config = self.load_yaml(stages_file)
            self.stages = config.get("stages", [])

        if modules_file.exists():
            config = self.load_yaml(modules_file)
            self.modules = config.get("modules", [])

        if projects_file.exists():
            config = self.load_yaml(projects_file)
            self.projects = config.get("projects", [])

        if datasets_file.exists():
            config = self.load_yaml(datasets_file)
            self.datasets = config.get("datasets", [])

    def validate_stages(self) -> bool:
        """验证stages.yaml"""
        print("🔍 验证 stages.yaml...")

        if not self.stages:
            self.errors.append("❌ stages.yaml 为空或未加载")
            return False

        required_fields = ["id", "name", "name_en", "priority", "level", "modules", "projects"]
        stage_ids: Set[str] = set()

        for stage in self.stages:
            stage_id = stage.get("id", "UNKNOWN")

            # 检查必填字段
            for field in required_fields:
                if field not in stage:
                    self.errors.append(f"❌ Stage {stage_id}: 缺少必填字段 '{field}'")

            # 检查ID唯一性
            if stage_id in stage_ids:
                self.errors.append(f"❌ Stage {stage_id}: ID重复")
            stage_ids.add(stage_id)

            # 检查优先级
            if stage.get("priority") not in ["P1", "P2", "P3"]:
                self.errors.append(f"❌ Stage {stage_id}: 优先级必须为P1/P2/P3")

            # 检查学习等级
            if stage.get("level") not in ["beginner", "intermediate", "advanced"]:
                self.errors.append(f"❌ Stage {stage_id}: 等级必须为beginner/intermediate/advanced")

        print(f"   ✅ 验证了 {len(self.stages)} 个阶段")
        return len(self.errors) == 0

    def validate_modules(self) -> bool:
        """验证modules.yaml"""
        print("🔍 验证 modules.yaml...")

        if not self.modules:
            self.errors.append("❌ modules.yaml 为空或未加载")
            return False

        required_fields = ["id", "stage_id", "name", "name_en", "order", "topics", "learning_materials"]
        module_ids: Set[str] = set()
        stage_ids = {s["id"] for s in self.stages}

        for module in self.modules:
            module_id = module.get("id", "UNKNOWN")
            stage_id = module.get("stage_id", "UNKNOWN")

            # 检查必填字段
            for field in required_fields:
                if field not in module:
                    self.errors.append(f"❌ Module {module_id}: 缺少必填字段 '{field}'")

            # 检查ID唯一性（按阶段）
            key = f"{stage_id}-{module_id}"
            if key in module_ids:
                self.errors.append(f"❌ Module {module_id} (Stage {stage_id}): ID重复")
            module_ids.add(key)

            # 检查stage_id引用
            if stage_id not in stage_ids:
                self.errors.append(f"❌ Module {module_id}: 引用了不存在的stage_id '{stage_id}'")

            # 检查学习材料
            if not isinstance(module.get("learning_materials", []), list):
                self.errors.append(f"❌ Module {module_id}: learning_materials必须是列表")

        print(f"   ✅ 验证了 {len(self.modules)} 个模块")
        return len(self.errors) == 0

    def validate_projects(self) -> bool:
        """验证projects.yaml"""
        print("🔍 验证 projects.yaml...")

        if not self.projects:
            self.errors.append("❌ projects.yaml 为空或未加载")
            return False

        required_fields = ["id", "stage_id", "name", "name_en", "industry", "order", "datasets", "techniques", "deliverables"]
        project_ids: Set[str] = set()
        stage_ids = {s["id"] for s in self.stages}

        for project in self.projects:
            project_id = project.get("id", "UNKNOWN")
            stage_id = project.get("stage_id", "UNKNOWN")

            # 检查必填字段
            for field in required_fields:
                if field not in project:
                    self.errors.append(f"❌ Project {project_id}: 缺少必填字段 '{field}'")

            # 检查ID唯一性（按阶段）
            key = f"{stage_id}-{project_id}"
            if key in project_ids:
                self.errors.append(f"❌ Project {project_id} (Stage {stage_id}): ID重复")
            project_ids.add(key)

            # 检查stage_id引用
            if stage_id not in stage_ids:
                self.errors.append(f"❌ Project {project_id}: 引用了不存在的stage_id '{stage_id}'")

            # 检查difficulty
            if project.get("difficulty") and project["difficulty"] not in ["beginner", "intermediate", "advanced"]:
                self.errors.append(f"❌ Project {project_id}: 难度必须为beginner/intermediate/advanced")

        print(f"   ✅ 验证了 {len(self.projects)} 个项目")
        return len(self.errors) == 0

    def validate_datasets(self) -> bool:
        """验证datasets.yaml"""
        print("🔍 验证 datasets.yaml...")

        if not self.datasets:
            self.errors.append("❌ datasets.yaml 为空或未加载")
            return False

        required_fields = ["id", "project_id", "stage_id", "name", "name_en", "source", "files"]
        dataset_ids: Set[str] = set()
        stage_ids = {s["id"] for s in self.stages}
        project_ids = {f"{p['stage_id']}-{p['id']}" for p in self.projects}

        for dataset in self.datasets:
            dataset_id = dataset.get("id", "UNKNOWN")
            project_id = dataset.get("project_id", "UNKNOWN")
            stage_id = dataset.get("stage_id", "UNKNOWN")

            # 检查必填字段
            for field in required_fields:
                if field not in dataset:
                    self.errors.append(f"❌ Dataset {dataset_id}: 缺少必填字段 '{field}'")

            # 检查ID唯一性
            if dataset_id in dataset_ids:
                self.errors.append(f"❌ Dataset {dataset_id}: ID重复")
            dataset_ids.add(dataset_id)

            # 检查stage_id引用
            if stage_id not in stage_ids:
                self.errors.append(f"❌ Dataset {dataset_id}: 引用了不存在的stage_id '{stage_id}'")

            # 检查project_id引用
            key = f"{stage_id}-{project_id}"
            if key not in project_ids:
                self.warnings.append(f"⚠️  Dataset {dataset_id}: 引用的project_id '{project_id}'可能不存在")

            # 检查文件列表
            if not isinstance(dataset.get("files", []), list) or len(dataset["files"]) == 0:
                self.errors.append(f"❌ Dataset {dataset_id}: files必须是非空列表")

        print(f"   ✅ 验证了 {len(self.datasets)} 个数据集")
        return len(self.errors) == 0

    def validate_cross_references(self) -> bool:
        """验证跨实体引用"""
        print("🔍 验证跨实体引用...")

        # 验证stages中的modules和projects引用
        module_ids_by_stage = {stage["id"]: [m["id"] for m in self.modules if m["stage_id"] == stage["id"]] for stage in self.stages}
        project_ids_by_stage = {stage["id"]: [p["id"] for p in self.projects if p["stage_id"] == stage["id"]] for stage in self.stages}

        for stage in self.stages:
            stage_id = stage["id"]

            # 检查modules引用
            for module_id in stage.get("modules", []):
                if module_id not in module_ids_by_stage.get(stage_id, []):
                    self.errors.append(f"❌ Stage {stage_id}: 引用了不存在的module '{module_id}'")

            # 检查projects引用
            for project_id in stage.get("projects", []):
                if project_id not in project_ids_by_stage.get(stage_id, []):
                    self.errors.append(f"❌ Stage {stage_id}: 引用了不存在的project '{project_id}'")

        # 验证projects中的datasets引用
        dataset_ids = {ds["id"] for ds in self.datasets}

        for project in self.projects:
            project_id = project["id"]
            stage_id = project["stage_id"]

            for dataset_id in project.get("datasets", []):
                if dataset_id not in dataset_ids:
                    self.errors.append(f"❌ Project {project_id}: 引用了不存在的dataset '{dataset_id}'")

        print(f"   ✅ 跨实体引用验证完成")
        return len(self.errors) == 0

    def validate_all(self, verbose: bool = False) -> bool:
        """验证所有实体"""
        print("=" * 70)
        print("🔍 实体配置验证 (Entity Configuration Validation)")
        print("=" * 70)
        print(f"配置目录: {self.config_dir}")
        print()

        # 加载所有实体
        self.load_all_entities()

        # 验证各个实体
        self.validate_stages()
        self.validate_modules()
        self.validate_projects()
        self.validate_datasets()
        self.validate_cross_references()

        # 打印结果
        print("\n" + "=" * 70)
        print("📊 验证结果")
        print("=" * 70)

        if self.errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"   {error}")

        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"   {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ 所有实体配置验证通过！")

        print("=" * 70)

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="验证YAML实体配置文件"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="配置文件目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    args = parser.parse_args()

    # 检查目录是否存在
    if not args.config_dir.exists():
        print(f"❌ 配置目录不存在: {args.config_dir}")
        sys.exit(1)

    # 初始化验证器
    validator = EntityValidator(config_dir=args.config_dir)

    # 执行验证
    success = validator.validate_all(verbose=args.verbose)

    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
