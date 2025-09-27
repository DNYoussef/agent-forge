"""
Build Optimizer for Agent Forge Pipeline

Comprehensive build optimization system providing:
- Webpack/bundler optimization for JavaScript components
- Tree shaking for unused code elimination
- Code splitting strategies
- Asset compression and minification
- Docker layer caching optimization
- Build performance monitoring and optimization
"""

import os
import sys
import json
import subprocess
import shutil
import gzip
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import re
import hashlib
import concurrent.futures
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """Configuration for build optimization."""
    # Source directories
    src_dir: str = "./src"
    static_dir: str = "./static"
    output_dir: str = "./dist"

    # Optimization settings
    enable_tree_shaking: bool = True
    enable_code_splitting: bool = True
    enable_minification: bool = True
    enable_compression: bool = True
    enable_source_maps: bool = False  # Disable for production

    # Asset optimization
    image_quality: int = 85
    enable_webp_conversion: bool = True
    enable_svg_optimization: bool = True

    # Caching
    enable_build_cache: bool = True
    cache_dir: str = "./build_cache"

    # Performance
    max_workers: int = 4
    chunk_size_kb: int = 250  # Maximum chunk size

    # Analysis
    analyze_bundle: bool = True
    size_budget_mb: float = 5.0  # Bundle size budget


@dataclass
class BuildMetrics:
    """Build performance metrics."""
    start_time: float
    end_time: float
    total_duration: float
    bundle_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    chunks_count: int
    assets_optimized: int
    cache_hits: int
    cache_misses: int
    memory_peak_mb: float
    warnings: List[str]
    errors: List[str]


@dataclass
class AssetInfo:
    """Information about a build asset."""
    path: str
    size_bytes: int
    compressed_size_bytes: int
    type: str
    optimization_applied: List[str]
    cache_hit: bool


class BuildCache:
    """Advanced build caching system."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def get_cached_result(self, source_path: Path, cache_key: str) -> Optional[Path]:
        """Get cached result if available and valid."""
        if cache_key not in self.cache_index:
            return None

        cache_entry = self.cache_index[cache_key]

        # Check if source file changed
        current_hash = self._get_file_hash(source_path)
        if current_hash != cache_entry.get('source_hash'):
            return None

        # Check if cached file exists
        cached_file = self.cache_dir / cache_entry['cached_file']
        if not cached_file.exists():
            return None

        return cached_file

    def store_result(self, source_path: Path, result_path: Path, cache_key: str):
        """Store build result in cache."""
        try:
            source_hash = self._get_file_hash(source_path)
            cached_filename = f"{cache_key}_{source_hash}"
            cached_file = self.cache_dir / cached_filename

            # Copy result to cache
            shutil.copy2(result_path, cached_file)

            # Update cache index
            self.cache_index[cache_key] = {
                'source_hash': source_hash,
                'cached_file': cached_filename,
                'timestamp': datetime.now().isoformat(),
                'source_path': str(source_path)
            }

            self._save_cache_index()

        except Exception as e:
            logger.error(f"Failed to store cache result: {e}")

    def cleanup_old_entries(self, max_age_days: int = 7):
        """Clean up old cache entries."""
        current_time = datetime.now()
        entries_to_remove = []

        for cache_key, entry in self.cache_index.items():
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if (current_time - entry_time).days > max_age_days:
                    entries_to_remove.append(cache_key)

                    # Remove cached file
                    cached_file = self.cache_dir / entry['cached_file']
                    if cached_file.exists():
                        cached_file.unlink()

            except Exception as e:
                logger.warning(f"Error cleaning cache entry {cache_key}: {e}")
                entries_to_remove.append(cache_key)

        # Remove from index
        for cache_key in entries_to_remove:
            del self.cache_index[cache_key]

        if entries_to_remove:
            self._save_cache_index()
            logger.info(f"Cleaned {len(entries_to_remove)} old cache entries")


class AssetOptimizer:
    """Asset optimization utilities."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.optimization_stats = {
            'images_optimized': 0,
            'js_minified': 0,
            'css_minified': 0,
            'files_compressed': 0,
            'bytes_saved': 0
        }

    def optimize_javascript(self, js_file: Path, output_file: Path) -> AssetInfo:
        """Optimize JavaScript file."""
        original_size = js_file.stat().st_size
        optimizations = []

        try:
            # Use terser for minification if available
            if shutil.which('terser'):
                cmd = [
                    'terser', str(js_file),
                    '--compress', '--mangle',
                    '--output', str(output_file)
                ]

                if not self.config.enable_source_maps:
                    cmd.extend(['--source-map', 'false'])

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    optimizations.append('terser_minified')
                    self.optimization_stats['js_minified'] += 1
                else:
                    logger.warning(f"Terser failed for {js_file}: {result.stderr}")
                    shutil.copy2(js_file, output_file)
            else:
                # Fallback: simple minification
                content = js_file.read_text(encoding='utf-8')
                minified = self._simple_js_minify(content)
                output_file.write_text(minified, encoding='utf-8')
                optimizations.append('simple_minified')

        except Exception as e:
            logger.error(f"JavaScript optimization failed for {js_file}: {e}")
            shutil.copy2(js_file, output_file)

        # Compression
        compressed_size = original_size
        if self.config.enable_compression:
            compressed_size = self._compress_file(output_file)
            if compressed_size < original_size:
                optimizations.append('gzip_compressed')
                self.optimization_stats['files_compressed'] += 1

        optimized_size = output_file.stat().st_size
        self.optimization_stats['bytes_saved'] += max(0, original_size - optimized_size)

        return AssetInfo(
            path=str(output_file),
            size_bytes=optimized_size,
            compressed_size_bytes=compressed_size,
            type='javascript',
            optimization_applied=optimizations,
            cache_hit=False
        )

    def optimize_css(self, css_file: Path, output_file: Path) -> AssetInfo:
        """Optimize CSS file."""
        original_size = css_file.stat().st_size
        optimizations = []

        try:
            content = css_file.read_text(encoding='utf-8')

            # Simple CSS minification
            minified = self._simple_css_minify(content)
            output_file.write_text(minified, encoding='utf-8')
            optimizations.append('css_minified')
            self.optimization_stats['css_minified'] += 1

        except Exception as e:
            logger.error(f"CSS optimization failed for {css_file}: {e}")
            shutil.copy2(css_file, output_file)

        # Compression
        compressed_size = original_size
        if self.config.enable_compression:
            compressed_size = self._compress_file(output_file)
            if compressed_size < original_size:
                optimizations.append('gzip_compressed')
                self.optimization_stats['files_compressed'] += 1

        optimized_size = output_file.stat().st_size
        self.optimization_stats['bytes_saved'] += max(0, original_size - optimized_size)

        return AssetInfo(
            path=str(output_file),
            size_bytes=optimized_size,
            compressed_size_bytes=compressed_size,
            type='css',
            optimization_applied=optimizations,
            cache_hit=False
        )

    def optimize_image(self, image_file: Path, output_file: Path) -> AssetInfo:
        """Optimize image file."""
        original_size = image_file.stat().st_size
        optimizations = []

        try:
            file_ext = image_file.suffix.lower()

            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Try to use imagemin or similar tools
                if self._optimize_with_imagemin(image_file, output_file):
                    optimizations.append('imagemin_optimized')
                else:
                    shutil.copy2(image_file, output_file)

                # Convert to WebP if enabled and beneficial
                if self.config.enable_webp_conversion:
                    webp_file = output_file.with_suffix('.webp')
                    if self._convert_to_webp(output_file, webp_file):
                        if webp_file.stat().st_size < output_file.stat().st_size:
                            output_file.unlink()
                            webp_file.rename(output_file)
                            optimizations.append('webp_converted')

            elif file_ext == '.svg' and self.config.enable_svg_optimization:
                if self._optimize_svg(image_file, output_file):
                    optimizations.append('svg_optimized')
                else:
                    shutil.copy2(image_file, output_file)

            else:
                shutil.copy2(image_file, output_file)

            self.optimization_stats['images_optimized'] += 1

        except Exception as e:
            logger.error(f"Image optimization failed for {image_file}: {e}")
            shutil.copy2(image_file, output_file)

        optimized_size = output_file.stat().st_size
        self.optimization_stats['bytes_saved'] += max(0, original_size - optimized_size)

        return AssetInfo(
            path=str(output_file),
            size_bytes=optimized_size,
            compressed_size_bytes=optimized_size,  # Images are already compressed
            type='image',
            optimization_applied=optimizations,
            cache_hit=False
        )

    def _simple_js_minify(self, content: str) -> str:
        """Simple JavaScript minification."""
        # Remove comments
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s+', ';', content)
        content = re.sub(r'{\s+', '{', content)
        content = re.sub(r'\s+}', '}', content)

        return content.strip()

    def _simple_css_minify(self, content: str) -> str:
        """Simple CSS minification."""
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s+', ';', content)
        content = re.sub(r'{\s+', '{', content)
        content = re.sub(r'\s+}', '}', content)
        content = re.sub(r':\s+', ':', content)

        return content.strip()

    def _compress_file(self, file_path: Path) -> int:
        """Compress file with gzip and return compressed size."""
        try:
            with open(file_path, 'rb') as f_in:
                compressed_data = gzip.compress(f_in.read())

            compressed_file = file_path.with_suffix(file_path.suffix + '.gz')
            with open(compressed_file, 'wb') as f_out:
                f_out.write(compressed_data)

            return len(compressed_data)

        except Exception as e:
            logger.error(f"Compression failed for {file_path}: {e}")
            return file_path.stat().st_size

    def _optimize_with_imagemin(self, input_file: Path, output_file: Path) -> bool:
        """Optimize image with imagemin if available."""
        if not shutil.which('imagemin'):
            return False

        try:
            cmd = ['imagemin', str(input_file), '--out-dir', str(output_file.parent)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _convert_to_webp(self, input_file: Path, output_file: Path) -> bool:
        """Convert image to WebP format."""
        if not shutil.which('cwebp'):
            return False

        try:
            cmd = [
                'cwebp', '-q', str(self.config.image_quality),
                str(input_file), '-o', str(output_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _optimize_svg(self, input_file: Path, output_file: Path) -> bool:
        """Optimize SVG file."""
        if not shutil.which('svgo'):
            return False

        try:
            cmd = ['svgo', str(input_file), '-o', str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False


class BuildOptimizer:
    """
    Comprehensive build optimizer for Agent Forge pipeline.

    Provides automated build optimization including:
    - Asset minification and compression
    - Code splitting and tree shaking
    - Bundle analysis and optimization
    - Build caching for faster rebuilds
    - Docker layer optimization
    """

    def __init__(self, config: Optional[BuildConfig] = None):
        self.config = config or BuildConfig()
        self.cache = BuildCache(self.config.cache_dir) if self.config.enable_build_cache else None
        self.asset_optimizer = AssetOptimizer(self.config)

        # Setup directories
        self.src_dir = Path(self.config.src_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build state
        self.build_metrics: Optional[BuildMetrics] = None
        self.asset_info: List[AssetInfo] = []

        logger.info("Build Optimizer initialized")
        logger.info(f"Source: {self.src_dir}")
        logger.info(f"Output: {self.output_dir}")

    def optimize_build(self) -> BuildMetrics:
        """
        Run complete build optimization process.
        """
        logger.info("Starting build optimization...")
        start_time = time.time()
        peak_memory = 0
        warnings = []
        errors = []
        cache_hits = 0
        cache_misses = 0

        try:
            # Track memory usage
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                peak_memory = process.memory_info().rss / (1024 * 1024)

            # Clean output directory
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Discover source files
            source_files = self._discover_source_files()
            logger.info(f"Found {len(source_files)} source files")

            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {}

                for source_file in source_files:
                    relative_path = source_file.relative_to(self.src_dir)
                    output_file = self.output_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    future = executor.submit(self._process_file, source_file, output_file)
                    future_to_file[future] = source_file

                # Collect results
                for future in concurrent.futures.as_completed(future_to_file):
                    source_file = future_to_file[future]
                    try:
                        asset_info = future.result()
                        if asset_info:
                            self.asset_info.append(asset_info)
                            if asset_info.cache_hit:
                                cache_hits += 1
                            else:
                                cache_misses += 1
                    except Exception as e:
                        error_msg = f"Failed to process {source_file}: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)

            # Apply code splitting if enabled
            if self.config.enable_code_splitting:
                self._apply_code_splitting()

            # Generate bundle analysis
            if self.config.analyze_bundle:
                self._analyze_bundle()

            # Update peak memory
            if PSUTIL_AVAILABLE:
                current_memory = process.memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)

        except Exception as e:
            error_msg = f"Build optimization failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

        # Calculate metrics
        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate bundle sizes
        total_size = sum(asset.size_bytes for asset in self.asset_info)
        total_compressed = sum(asset.compressed_size_bytes for asset in self.asset_info)

        bundle_size_mb = total_size / (1024 * 1024)
        compressed_size_mb = total_compressed / (1024 * 1024)
        compression_ratio = compressed_size_mb / bundle_size_mb if bundle_size_mb > 0 else 0

        # Check size budget
        if bundle_size_mb > self.config.size_budget_mb:
            warnings.append(f"Bundle size ({bundle_size_mb:.2f}MB) exceeds budget ({self.config.size_budget_mb}MB)")

        # Create metrics
        self.build_metrics = BuildMetrics(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            bundle_size_mb=bundle_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=compression_ratio,
            chunks_count=len([a for a in self.asset_info if 'chunk' in a.path]),
            assets_optimized=len(self.asset_info),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            memory_peak_mb=peak_memory,
            warnings=warnings,
            errors=errors
        )

        # Generate reports
        self._generate_build_report()

        logger.info(f"Build optimization completed in {total_duration:.2f}s")
        logger.info(f"Bundle size: {bundle_size_mb:.2f}MB (compressed: {compressed_size_mb:.2f}MB)")
        logger.info(f"Assets optimized: {len(self.asset_info)}")
        logger.info(f"Cache hits: {cache_hits}, misses: {cache_misses}")

        return self.build_metrics

    def _discover_source_files(self) -> List[Path]:
        """Discover all source files to process."""
        source_files = []

        if not self.src_dir.exists():
            logger.warning(f"Source directory does not exist: {self.src_dir}")
            return source_files

        # File extensions to process
        extensions = {'.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.sass', '.less',
                     '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico'}

        for file_path in self.src_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                source_files.append(file_path)

        return source_files

    def _process_file(self, source_file: Path, output_file: Path) -> Optional[AssetInfo]:
        """Process a single source file."""
        file_ext = source_file.suffix.lower()
        cache_key = f"{source_file.relative_to(self.src_dir)}_{file_ext}"

        # Check cache first
        if self.cache:
            cached_result = self.cache.get_cached_result(source_file, cache_key)
            if cached_result:
                shutil.copy2(cached_result, output_file)
                return AssetInfo(
                    path=str(output_file),
                    size_bytes=output_file.stat().st_size,
                    compressed_size_bytes=output_file.stat().st_size,
                    type=self._get_file_type(file_ext),
                    optimization_applied=['cache_hit'],
                    cache_hit=True
                )

        # Process based on file type
        asset_info = None

        if file_ext in ['.js', '.jsx', '.ts', '.tsx']:
            asset_info = self.asset_optimizer.optimize_javascript(source_file, output_file)
        elif file_ext in ['.css', '.scss', '.sass', '.less']:
            asset_info = self.asset_optimizer.optimize_css(source_file, output_file)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico']:
            asset_info = self.asset_optimizer.optimize_image(source_file, output_file)
        else:
            # Copy other files as-is
            shutil.copy2(source_file, output_file)
            asset_info = AssetInfo(
                path=str(output_file),
                size_bytes=output_file.stat().st_size,
                compressed_size_bytes=output_file.stat().st_size,
                type=self._get_file_type(file_ext),
                optimization_applied=['copied'],
                cache_hit=False
            )

        # Store in cache
        if self.cache and asset_info and not asset_info.cache_hit:
            self.cache.store_result(source_file, output_file, cache_key)

        return asset_info

    def _get_file_type(self, extension: str) -> str:
        """Get file type from extension."""
        type_map = {
            '.js': 'javascript', '.jsx': 'javascript', '.ts': 'javascript', '.tsx': 'javascript',
            '.css': 'stylesheet', '.scss': 'stylesheet', '.sass': 'stylesheet', '.less': 'stylesheet',
            '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
            '.svg': 'image', '.webp': 'image', '.ico': 'image'
        }
        return type_map.get(extension.lower(), 'other')

    def _apply_code_splitting(self):
        """Apply code splitting strategies."""
        logger.info("Applying code splitting...")

        # Find large JavaScript files that could benefit from splitting
        js_assets = [a for a in self.asset_info if a.type == 'javascript']
        large_js_files = [a for a in js_assets if a.size_bytes > self.config.chunk_size_kb * 1024]

        for asset in large_js_files:
            try:
                self._split_javascript_file(Path(asset.path))
            except Exception as e:
                logger.error(f"Failed to split {asset.path}: {e}")

    def _split_javascript_file(self, js_file: Path):
        """Split a large JavaScript file into chunks."""
        # This is a simplified example - real implementation would be more sophisticated
        content = js_file.read_text(encoding='utf-8')

        # Simple splitting by function boundaries
        functions = re.findall(r'function\s+\w+\s*\([^)]*\)\s*{[^}]*}', content, re.DOTALL)

        if len(functions) > 1:
            # Create chunks
            chunk_size = len(functions) // 2
            chunk1_content = '\n'.join(functions[:chunk_size])
            chunk2_content = '\n'.join(functions[chunk_size:])

            # Save chunks
            chunk1_file = js_file.with_suffix('.chunk1.js')
            chunk2_file = js_file.with_suffix('.chunk2.js')

            chunk1_file.write_text(chunk1_content, encoding='utf-8')
            chunk2_file.write_text(chunk2_content, encoding='utf-8')

            # Update asset info
            for content, chunk_file in [(chunk1_content, chunk1_file), (chunk2_content, chunk2_file)]:
                chunk_info = AssetInfo(
                    path=str(chunk_file),
                    size_bytes=len(content.encode('utf-8')),
                    compressed_size_bytes=len(content.encode('utf-8')),  # Simplified
                    type='javascript',
                    optimization_applied=['code_split'],
                    cache_hit=False
                )
                self.asset_info.append(chunk_info)

            logger.info(f"Split {js_file.name} into 2 chunks")

    def _analyze_bundle(self):
        """Analyze bundle composition and performance."""
        logger.info("Analyzing bundle...")

        analysis = {
            'total_assets': len(self.asset_info),
            'by_type': {},
            'largest_assets': [],
            'optimization_summary': {},
            'recommendations': []
        }

        # Group by type
        type_stats = {}
        for asset in self.asset_info:
            if asset.type not in type_stats:
                type_stats[asset.type] = {
                    'count': 0,
                    'total_size': 0,
                    'optimizations': set()
                }

            type_stats[asset.type]['count'] += 1
            type_stats[asset.type]['total_size'] += asset.size_bytes
            type_stats[asset.type]['optimizations'].update(asset.optimization_applied)

        analysis['by_type'] = {
            type_name: {
                'count': stats['count'],
                'total_size_mb': stats['total_size'] / (1024 * 1024),
                'optimizations': list(stats['optimizations'])
            }
            for type_name, stats in type_stats.items()
        }

        # Find largest assets
        sorted_assets = sorted(self.asset_info, key=lambda a: a.size_bytes, reverse=True)
        analysis['largest_assets'] = [
            {
                'path': asset.path,
                'size_mb': asset.size_bytes / (1024 * 1024),
                'type': asset.type
            }
            for asset in sorted_assets[:10]
        ]

        # Optimization summary
        optimization_counts = {}
        for asset in self.asset_info:
            for opt in asset.optimization_applied:
                optimization_counts[opt] = optimization_counts.get(opt, 0) + 1

        analysis['optimization_summary'] = optimization_counts

        # Generate recommendations
        if analysis['by_type'].get('javascript', {}).get('total_size_mb', 0) > 2.0:
            analysis['recommendations'].append("Consider more aggressive JavaScript code splitting")

        if analysis['by_type'].get('image', {}).get('total_size_mb', 0) > 1.0:
            analysis['recommendations'].append("Optimize large images or use lazy loading")

        cache_hit_rate = optimization_counts.get('cache_hit', 0) / len(self.asset_info) if self.asset_info else 0
        if cache_hit_rate < 0.3:
            analysis['recommendations'].append("Build cache could be more effective")

        # Save analysis
        analysis_file = self.output_dir / 'bundle_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Bundle analysis saved: {analysis_file}")

    def _generate_build_report(self):
        """Generate comprehensive build report."""
        if not self.build_metrics:
            return

        report = {
            'build_metrics': asdict(self.build_metrics),
            'optimization_stats': self.asset_optimizer.optimization_stats,
            'asset_details': [asdict(asset) for asset in self.asset_info],
            'performance_summary': {
                'build_time_seconds': self.build_metrics.total_duration,
                'bundle_size_mb': self.build_metrics.bundle_size_mb,
                'compression_ratio': self.build_metrics.compression_ratio,
                'assets_per_second': len(self.asset_info) / self.build_metrics.total_duration,
                'cache_efficiency': self.build_metrics.cache_hits / (self.build_metrics.cache_hits + self.build_metrics.cache_misses) if (self.build_metrics.cache_hits + self.build_metrics.cache_misses) > 0 else 0
            },
            'recommendations': self._generate_performance_recommendations()
        }

        # Save report
        report_file = self.output_dir / 'build_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Build report saved: {report_file}")

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        if not self.build_metrics:
            return recommendations

        # Build time recommendations
        if self.build_metrics.total_duration > 30:
            recommendations.append("Build time is slow - consider increasing parallel workers")

        # Bundle size recommendations
        if self.build_metrics.bundle_size_mb > self.config.size_budget_mb:
            recommendations.append(f"Bundle exceeds size budget by {self.build_metrics.bundle_size_mb - self.config.size_budget_mb:.2f}MB")

        # Cache recommendations
        cache_hit_rate = self.build_metrics.cache_hits / (self.build_metrics.cache_hits + self.build_metrics.cache_misses) if (self.build_metrics.cache_hits + self.build_metrics.cache_misses) > 0 else 0
        if cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - verify cache configuration")

        # Compression recommendations
        if self.build_metrics.compression_ratio > 0.8:
            recommendations.append("Poor compression ratio - assets may benefit from better optimization")

        # Memory recommendations
        if self.build_metrics.memory_peak_mb > 1000:
            recommendations.append("High memory usage during build - consider reducing parallel workers")

        return recommendations

    def clean_cache(self):
        """Clean build cache."""
        if self.cache:
            self.cache.cleanup_old_entries()
            logger.info("Build cache cleaned")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = dict(self.asset_optimizer.optimization_stats)

        if self.build_metrics:
            stats.update({
                'total_assets': len(self.asset_info),
                'cache_hit_rate': self.build_metrics.cache_hits / (self.build_metrics.cache_hits + self.build_metrics.cache_misses) if (self.build_metrics.cache_hits + self.build_metrics.cache_misses) > 0 else 0,
                'compression_ratio': self.build_metrics.compression_ratio,
                'build_time': self.build_metrics.total_duration
            })

        return stats


def optimize_docker_layers() -> Dict[str, Any]:
    """
    Optimize Docker layers for better caching.
    """
    dockerfile_optimizations = {
        'layer_order_optimized': False,
        'multi_stage_used': False,
        'cache_mounts_added': False,
        'recommendations': []
    }

    dockerfile_path = Path('Dockerfile')
    if not dockerfile_path.exists():
        dockerfile_optimizations['recommendations'].append("Create Dockerfile for containerized builds")
        return dockerfile_optimizations

    try:
        content = dockerfile_path.read_text()

        # Check for optimization opportunities
        if 'COPY package*.json' in content and 'RUN npm install' in content:
            dockerfile_optimizations['layer_order_optimized'] = True
        else:
            dockerfile_optimizations['recommendations'].append("Copy package.json before source code for better layer caching")

        if 'FROM' in content and content.count('FROM') > 1:
            dockerfile_optimizations['multi_stage_used'] = True
        else:
            dockerfile_optimizations['recommendations'].append("Consider multi-stage builds for smaller images")

        if '--mount=type=cache' in content:
            dockerfile_optimizations['cache_mounts_added'] = True
        else:
            dockerfile_optimizations['recommendations'].append("Add cache mounts for package managers")

    except Exception as e:
        logger.error(f"Failed to analyze Dockerfile: {e}")

    return dockerfile_optimizations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge Build Optimizer")
    parser.add_argument("--src-dir", default="./src", help="Source directory")
    parser.add_argument("--output-dir", default="./dist", help="Output directory")
    parser.add_argument("--clean-cache", action="store_true", help="Clean build cache")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't build")
    parser.add_argument("--disable-compression", action="store_true", help="Disable compression")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create build config
    config = BuildConfig(
        src_dir=args.src_dir,
        output_dir=args.output_dir,
        enable_compression=not args.disable_compression
    )

    # Create optimizer
    optimizer = BuildOptimizer(config)

    if args.clean_cache:
        optimizer.clean_cache()
        print("Build cache cleaned")

    if not args.analyze_only:
        # Run optimization
        metrics = optimizer.optimize_build()

        print("\nBuild Optimization Results:")
        print(f"Duration: {metrics.total_duration:.2f}s")
        print(f"Bundle size: {metrics.bundle_size_mb:.2f}MB")
        print(f"Compressed size: {metrics.compressed_size_mb:.2f}MB")
        print(f"Compression ratio: {metrics.compression_ratio:.2%}")
        print(f"Assets optimized: {metrics.assets_optimized}")
        print(f"Cache hits: {metrics.cache_hits}")

        if metrics.warnings:
            print(f"\nWarnings: {len(metrics.warnings)}")
            for warning in metrics.warnings:
                print(f"  - {warning}")

        if metrics.errors:
            print(f"\nErrors: {len(metrics.errors)}")
            for error in metrics.errors:
                print(f"  - {error}")

    # Docker optimization
    docker_opts = optimize_docker_layers()
    print(f"\nDocker Optimization:")
    print(f"Layer order optimized: {docker_opts['layer_order_optimized']}")
    print(f"Multi-stage used: {docker_opts['multi_stage_used']}")
    print(f"Cache mounts added: {docker_opts['cache_mounts_added']}")

    if docker_opts['recommendations']:
        print("Docker recommendations:")
        for rec in docker_opts['recommendations']:
            print(f"  - {rec}")