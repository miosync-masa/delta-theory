#!/usr/bin/env python3
"""
FatigueData-AM2022 Upstash Redis API
=====================================

δ理論検証用 疲労データベースAPI

Usage:
    from fatigue_redis_api import FatigueDB
    
    db = FatigueDB()
    
    # 材料一覧
    materials = db.list_materials()
    
    # S-Nデータ取得
    ti64_sn = db.get_sn('Ti-6Al-4V')
    
    # R=-1 のデータのみ
    ti64_r1 = db.get_sn('Ti-6Al-4V', R=-1.0)
    
    # σ_y付きデータのみ
    ti64_sy = db.get_sn('Ti-6Al-4V', with_sigma_y=True)
    
    # da/dN データ
    dadn = db.get_dadn('Ti-6Al-4V')

Author: 環 & ご主人さま
Date: 2026-02-02
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os

try:
    from upstash_redis import Redis
except ImportError:
    raise ImportError("pip install upstash-redis --break-system-packages")


# =============================================================================
# Configuration
# =============================================================================
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

class FatigueDB:
    def __init__(self, url: str = None, token: str = None):
        _url = url or UPSTASH_URL
        _token = token or UPSTASH_TOKEN
        if not _url or not _token:
            raise ValueError("UPSTASH_URL and UPSTASH_TOKEN required! Set env vars or pass directly.")
        self.redis = Redis(url=_url, token=_token)
        
# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class SNPoint:
    """S-N データポイント"""
    N: float           # 破断サイクル数
    S: float           # 応力振幅 [MPa]
    runout: bool       # ランアウト（未破断）フラグ
    R: Optional[float] # 応力比
    sigma_y: Optional[float]   # 降伏応力 [MPa]
    sigma_uts: Optional[float] # 引張強さ [MPa]
    doi: Optional[str] # 出典DOI


@dataclass  
class ENPoint:
    """ε-N データポイント"""
    N: float           # 破断サイクル数
    e: float           # ひずみ振幅
    runout: bool       # ランアウト
    R: Optional[float]
    sigma_y: Optional[float]
    E: Optional[float] # ヤング率 [GPa]
    doi: Optional[str]


@dataclass
class DaDNPoint:
    """da/dN-ΔK データポイント"""
    dK: float          # 応力拡大係数範囲 [MPa√m]
    dadn: float        # き裂進展速度 [m/cycle]
    R: Optional[float]
    doi: Optional[str]


# =============================================================================
# Main API Class
# =============================================================================
class FatigueDB:
    """FatigueData-AM2022 Upstash Redis API"""
    
    def __init__(self, url: str = UPSTASH_URL, token: str = UPSTASH_TOKEN):
        """Initialize connection to Upstash Redis"""
        self.redis = Redis(url=url, token=token)
        self._meta = None
        self._materials = None
    
    # -------------------------------------------------------------------------
    # Meta & Materials
    # -------------------------------------------------------------------------
    
    def get_meta(self) -> Dict[str, Any]:
        """データベースメタ情報"""
        if self._meta is None:
            self._meta = json.loads(self.redis.get('fatigue:meta'))
        return self._meta
    
    def list_materials(self, sort_by: str = 'sn_count') -> List[Dict]:
        """材料一覧を取得
        
        Args:
            sort_by: ソートキー ('sn_count', 'en_count', 'dadn_count', 'name')
        
        Returns:
            材料リスト [{name, sn_count, en_count, dadn_count, sigma_y_min, sigma_y_max}, ...]
        """
        if self._materials is None:
            self._materials = json.loads(self.redis.get('fatigue:materials'))
        
        result = []
        for name, info in self._materials.items():
            result.append({'name': name, **info})
        
        if sort_by == 'name':
            result.sort(key=lambda x: x['name'])
        elif sort_by in ['sn_count', 'en_count', 'dadn_count']:
            result.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
        
        return result
    
    def get_material_info(self, material: str) -> Optional[Dict]:
        """特定材料の情報"""
        if self._materials is None:
            self.list_materials()
        return self._materials.get(material)
    
    # -------------------------------------------------------------------------
    # S-N Data
    # -------------------------------------------------------------------------
    
    def get_sn(
        self, 
        material: str,
        R: Optional[float] = None,
        with_sigma_y: bool = False,
        as_dataclass: bool = False
    ) -> List[Dict | SNPoint]:
        """S-Nデータを取得
        
        Args:
            material: 材料名 (e.g., 'Ti-6Al-4V', '316L')
            R: 応力比でフィルタ (e.g., -1.0, 0.1)
            with_sigma_y: σ_yが存在するデータのみ
            as_dataclass: SNPointクラスで返す
        
        Returns:
            S-Nデータリスト
        """
        key = f'fatigue:sn:{material}'
        raw = self.redis.get(key)
        if raw is None:
            return []
        
        data = json.loads(raw)
        
        # フィルタ
        if R is not None:
            data = [d for d in data if d.get('R') == R]
        if with_sigma_y:
            data = [d for d in data if d.get('sigma_y') is not None]
        
        if as_dataclass:
            return [SNPoint(
                N=d['N'], S=d['S'], runout=bool(d.get('runout', 0)),
                R=d.get('R'), sigma_y=d.get('sigma_y'), 
                sigma_uts=d.get('sigma_uts'), doi=d.get('doi')
            ) for d in data]
        
        return data
    
    def get_sn_for_delta(
        self,
        material: str,
        R: float = -1.0
    ) -> List[Dict]:
        """δ理論検証用にr = σ_a/σ_y を計算済みのデータを取得
        
        Args:
            material: 材料名
            R: 応力比 (default: -1.0 for fully reversed)
        
        Returns:
            [{N, S, sigma_y, r, runout}, ...]
        """
        data = self.get_sn(material, R=R, with_sigma_y=True)
        
        result = []
        for d in data:
            sigma_y = d['sigma_y']
            # R=-1 の場合、S = σ_max = 2 * σ_a なので σ_a = S/2... ではない
            # 実際は S が応力振幅として記録されていることが多い
            # データセットの定義を確認: S は stress amplitude or max stress?
            # AM2022では Sは応力振幅（amplitude）として扱う
            sigma_a = d['S']  # stress amplitude [MPa]
            r = sigma_a / sigma_y
            
            result.append({
                'N': d['N'],
                'S': sigma_a,
                'sigma_y': sigma_y,
                'r': r,
                'runout': d.get('runout', 0),
                'doi': d.get('doi')
            })
        
        return result
    
    # -------------------------------------------------------------------------
    # ε-N Data
    # -------------------------------------------------------------------------
    
    def get_en(
        self,
        material: str,
        R: Optional[float] = None,
        as_dataclass: bool = False
    ) -> List[Dict | ENPoint]:
        """ε-Nデータを取得"""
        key = f'fatigue:en:{material}'
        raw = self.redis.get(key)
        if raw is None:
            return []
        
        data = json.loads(raw)
        
        if R is not None:
            data = [d for d in data if d.get('R') == R]
        
        if as_dataclass:
            return [ENPoint(
                N=d['N'], e=d['e'], runout=bool(d.get('runout', 0)),
                R=d.get('R'), sigma_y=d.get('sigma_y'),
                E=d.get('E'), doi=d.get('doi')
            ) for d in data]
        
        return data
    
    # -------------------------------------------------------------------------
    # da/dN-ΔK Data
    # -------------------------------------------------------------------------
    
    def get_dadn(
        self,
        material: str,
        R: Optional[float] = None,
        as_dataclass: bool = False
    ) -> List[Dict | DaDNPoint]:
        """da/dN-ΔKデータを取得（チャンク対応）"""
        # チャンクがあるか確認
        chunks_key = f'fatigue:dadn:{material}:chunks'
        n_chunks = self.redis.get(chunks_key)
        
        if n_chunks:
            # チャンクから読み込み
            data = []
            for i in range(int(n_chunks)):
                chunk = json.loads(self.redis.get(f'fatigue:dadn:{material}:chunk{i}'))
                data.extend(chunk)
        else:
            # 単一キー
            key = f'fatigue:dadn:{material}'
            raw = self.redis.get(key)
            if raw is None:
                return []
            data = json.loads(raw)
        
        if R is not None:
            data = [d for d in data if d.get('R') == R]
        
        if as_dataclass:
            return [DaDNPoint(
                dK=d['dK'], dadn=d['dadn'],
                R=d.get('R'), doi=d.get('doi')
            ) for d in data]
        
        return data
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def search_materials(self, query: str) -> List[str]:
        """材料名を検索"""
        if self._materials is None:
            self.list_materials()
        
        query_lower = query.lower()
        return [name for name in self._materials.keys() 
                if query_lower in name.lower()]
    
    def summary(self) -> str:
        """データベースサマリー"""
        meta = self.get_meta()
        return f"""
FatigueData-AM2022 @ Upstash Redis
===================================
Source: {meta['source']}
DOI: {meta['doi']}
License: {meta['license']}

Materials: {meta['total_materials']}
S-N points: {meta['total_sn_points']:,}
ε-N points: {meta['total_en_points']:,}
da/dN points: {meta['total_dadn_points']:,}
TOTAL: {meta['total_sn_points'] + meta['total_en_points'] + meta['total_dadn_points']:,}

Updated: {meta['updated']}
"""


# =============================================================================
# CLI
# =============================================================================
def build_cli():
    import argparse
    
    p = argparse.ArgumentParser(
        description='FatigueData-AM2022 Redis API - δ理論検証用'
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    
    # list
    sp_list = sub.add_parser('list', help='材料一覧')
    sp_list.add_argument('--top', type=int, default=20)
    sp_list.add_argument('--sort', choices=['sn_count', 'name'], default='sn_count')
    
    # search
    sp_search = sub.add_parser('search', help='材料検索')
    sp_search.add_argument('query', help='検索クエリ (例: Ti, 316, Al)')
    
    # info
    sp_info = sub.add_parser('info', help='材料詳細')
    sp_info.add_argument('material', help='材料名')
    
    # get-sn
    sp_sn = sub.add_parser('get-sn', help='S-Nデータ取得')
    sp_sn.add_argument('material', help='材料名')
    sp_sn.add_argument('--R', type=float, default=None, help='応力比フィルタ')
    sp_sn.add_argument('--with-sigma-y', action='store_true', help='σ_yありのみ')
    sp_sn.add_argument('--output', '-o', help='CSV出力ファイル')
    sp_sn.add_argument('--limit', type=int, default=20, help='表示件数')
    
    # delta
    sp_delta = sub.add_parser('delta', help='δ理論検証用データ (r計算済み)')
    sp_delta.add_argument('material', help='材料名')
    sp_delta.add_argument('--R', type=float, default=-1.0)
    sp_delta.add_argument('--output', '-o', help='CSV出力ファイル')
    
    # summary
    sub.add_parser('summary', help='DB全体サマリー')
    
    return p


def cmd_list(db, args):
    print(f"\n📦 材料一覧 (top {args.top}, sort={args.sort}):")
    print("-" * 70)
    print(f"{'Material':<25} {'S-N':>8} {'ε-N':>8} {'σ_y range':>20}")
    print("-" * 70)
    for mat in db.list_materials(sort_by=args.sort)[:args.top]:
        sy_min = mat.get('sigma_y_min', '-')
        sy_max = mat.get('sigma_y_max', '-')
        sy_range = f"{sy_min}-{sy_max}" if sy_min != '-' else '-'
        print(f"{mat['name']:<25} {mat['sn_count']:>8} {mat['en_count']:>8} {sy_range:>20}")


def cmd_search(db, args):
    results = db.search_materials(args.query)
    print(f"\n🔍 検索: '{args.query}' → {len(results)}件")
    for name in results:
        info = db.get_material_info(name)
        print(f"   {name}: S-N={info['sn_count']}, σ_y={info.get('sigma_y_min')}-{info.get('sigma_y_max')} MPa")


def cmd_info(db, args):
    info = db.get_material_info(args.material)
    if not info:
        print(f"❌ Material '{args.material}' not found")
        return
    
    print(f"\n📋 {args.material}")
    print("=" * 50)
    print(f"  S-N points:    {info['sn_count']}")
    print(f"  ε-N points:    {info['en_count']}")
    print(f"  da/dN points:  {info['dadn_count']}")
    print(f"  σ_y range:     {info.get('sigma_y_min')} - {info.get('sigma_y_max')} MPa")
    
    # R値の分布
    sn_data = db.get_sn(args.material)
    R_values = set(d.get('R') for d in sn_data if d.get('R') is not None)
    print(f"  R values:      {sorted(R_values)}")


def cmd_get_sn(db, args):
    data = db.get_sn(args.material, R=args.R, with_sigma_y=args.with_sigma_y)
    
    print(f"\n📊 {args.material} S-N data")
    if args.R is not None:
        print(f"   R = {args.R}")
    if args.with_sigma_y:
        print(f"   (σ_y required)")
    print(f"   Total: {len(data)} points")
    print("-" * 70)
    
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['N', 'S', 'R', 'sigma_y', 'sigma_uts', 'runout', 'doi'])
            w.writeheader()
            w.writerows(data)
        print(f"✅ Saved to {args.output}")
    else:
        print(f"{'N':>12} {'S [MPa]':>10} {'R':>6} {'σ_y':>10} {'runout':>8}")
        print("-" * 50)
        for d in data[:args.limit]:
            print(f"{d['N']:>12.0f} {d['S']:>10.1f} {d.get('R', '-'):>6} {d.get('sigma_y', '-'):>10} {d.get('runout', 0):>8}")
        if len(data) > args.limit:
            print(f"   ... and {len(data) - args.limit} more (use --output for full data)")


def cmd_delta(db, args):
    data = db.get_sn_for_delta(args.material, R=args.R)
    
    print(f"\n🔬 {args.material} δ-theory data (R={args.R})")
    print(f"   Total: {len(data)} points with σ_y")
    if data:
        print(f"   r range: {min(d['r'] for d in data):.4f} - {max(d['r'] for d in data):.4f}")
    print("-" * 70)
    
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['N', 'S', 'sigma_y', 'r', 'runout', 'doi'])
            w.writeheader()
            w.writerows(data)
        print(f"✅ Saved to {args.output}")
    else:
        print(f"{'N':>12} {'S [MPa]':>10} {'σ_y [MPa]':>12} {'r':>10} {'runout':>8}")
        print("-" * 60)
        for d in data[:20]:
            print(f"{d['N']:>12.0f} {d['S']:>10.1f} {d['sigma_y']:>12.1f} {d['r']:>10.4f} {d.get('runout', 0):>8}")
        if len(data) > 20:
            print(f"   ... and {len(data) - 20} more")


if __name__ == '__main__':
    parser = build_cli()
    args = parser.parse_args()
    
    db = FatigueDB()
    
    if args.cmd == 'summary':
        print(db.summary())
    elif args.cmd == 'list':
        cmd_list(db, args)
    elif args.cmd == 'search':
        cmd_search(db, args)
    elif args.cmd == 'info':
        cmd_info(db, args)
    elif args.cmd == 'get-sn':
        cmd_get_sn(db, args)
    elif args.cmd == 'delta':
        cmd_delta(db, args)
